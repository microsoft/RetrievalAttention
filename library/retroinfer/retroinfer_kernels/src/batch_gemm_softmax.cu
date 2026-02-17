#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <vector>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm.h"
#include "helper.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

#include "batch_gemm_softmax.h"


template<typename T>
void batch_gemm_softmax_impl(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha = 1.0,
    float beta = 0.0
) {
    /// GEMM types
    using ElementA = T;
    using ElementB = T;
    using ElementC = T;
    using ElementCompute = float;
    using ElementD = ElementC;
    /// Softmax types
    using ElementSoftmax = ElementC;
    using ElementSoftmaxCompute = float;
    using ElementNorm = float;
    using ElementSum = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentSoftmax = 128 / cutlass::sizeof_bits<ElementSoftmax>::value;

    using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    // ApplyShape for final Softmax.
    using ApplyShape = cutlass::MatrixShape<1, 1024>;
    static int const kStages = 4;

    /// Linear scaling operator
    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementCompute,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

    using BatchGemmSoftmax = cutlass::BatchGemmSoftmax<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC,
        ElementCompute,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueFunctorOp,
        kStages,
        ApplyShape,
        AlignmentA,
        AlignmentB,
        AlignmentSoftmax,
        ElementNorm,
        ElementSum,
        ElementSoftmax,
        ElementSoftmaxCompute
    >;

    using LayoutC = typename BatchGemmSoftmax::LayoutC;
    using LayoutN = typename BatchGemmSoftmax::LayoutN;
    using LayoutS = typename BatchGemmSoftmax::LayoutS;
    using MatrixCoord = typename LayoutC::TensorCoord;

    cutlass::gemm::GemmCoord problem = {m, n, k};

    int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // fixed rowmajor for norm and sum
    int64_t ldn = problem.m();
    int64_t lds = problem.m();

    int block_num = (problem.n() + BatchGemmSoftmax::ThreadblockShape::kN - 1) / BatchGemmSoftmax::ThreadblockShape::kN;

    typename BatchGemmSoftmax::Arguments args(
      problem,
      batch_count,
      {reinterpret_cast<ElementA *>(A.data_ptr()), lda},
      {reinterpret_cast<ElementB *>(B.data_ptr()), ldb},
      {nullptr, ldc},
      {reinterpret_cast<ElementD *>(D.data_ptr()), ldc},
      {
        ElementCompute(alpha),
        ElementCompute(beta)
      },
      {reinterpret_cast<ElementNorm *>(Norm.data_ptr()), ldn},
      {reinterpret_cast<ElementSum *>(Sum.data_ptr()), lds},
      {reinterpret_cast<ElementSoftmax *>(Softmax.data_ptr()), ldc},
      problem.m() * problem.k(),
      problem.k() * problem.n(),
      problem.m() * problem.n(),
      problem.m() * problem.n(),
      block_num * problem.m(),
      block_num * problem.m(),
      problem.m() * problem.n()
    );

    BatchGemmSoftmax batch_gemm_softmax;

    CUTLASS_CHECK(batch_gemm_softmax.initialize(args));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    CUTLASS_CHECK(batch_gemm_softmax(stream));
}

void batch_gemm_softmax(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha = 1.0,
    float beta = 0.0
) {
    if (A.dtype() == torch::kBFloat16) {
        batch_gemm_softmax_impl<cutlass::bfloat16_t>(
            A, B, D, Norm, Sum, Softmax,
            batch_count, m, n, k, alpha, beta
        );
    } else if (A.dtype() == torch::kFloat16) {
        batch_gemm_softmax_impl<cutlass::half_t>(
            A, B, D, Norm, Sum, Softmax,
            batch_count, m, n, k, alpha, beta
        );
    } else {
        TORCH_CHECK(false, "Only BFloat16 and Float16 dtypes are supported");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_gemm_softmax", &batch_gemm_softmax, "Batch GEMM Softmax (CUDA)",
        py::arg("A"), py::arg("B"), py::arg("D"),
        py::arg("Norm"), py::arg("Sum"), py::arg("Softmax"),
        py::arg("batch_count"), py::arg("m"), py::arg("n"), py::arg("k"),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f
    );
}