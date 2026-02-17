#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CHUNK_SIZE 8        // number of vectors for one chunk
#define DATA_BYTES 2        // input data type, 2 use bf16/fp16, 4 use fp32
#define PTYPE int2          // int4 for 16 Bytes, int2 for 8 Bytes

#if DATA_BYTES == 4
#define MYTPE uint32_t
#else
#define MYTPE uint16_t
#endif

#include "copy_kernel.cuh"


// gather copy keys, values and metadata in vector-level
void gather_copy_vectors(
    torch::Tensor& key_data,        // input, shape (groups, num_vectors, dim)
    torch::Tensor& key_buffer,      // output, shape (groups, buffer_size, dim)
    torch::Tensor& value_data,      // input, shape (groups, num_vectors, dim)
    torch::Tensor& value_buffer,    // output, shape (groups, buffer_size, dim)
    torch::Tensor& src_metadata,    // input, shape (groups, num_vectors)
    torch::Tensor& dst_metadata,    // output, shape (groups, buffer_size)
    
    torch::Tensor& data_offsets,    // input, shape (groups, index_size)

    int groups, 
    int data_length,                // num_vectors
    int buffer_length,              // buffer_size
    int index_size,                 // index_size
    int copy_start,                 // start vector index for copy
    int copy_num                    // number of vectors need to copy from data
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = copy_num * sizeof(int);

    PTYPE* key_data_ptr;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr;
    PTYPE* value_buffer_ptr;
    MYTPE* src_metadata_ptr;
    MYTPE* dst_metadata_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
    src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<float>());
    dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<float>());
#else
    if (key_data.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
        src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<at::Half>());
        dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
        src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<at::BFloat16>());
        dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<at::BFloat16>());
    }
#endif
    
    int64_t* data_offsets_ptr = reinterpret_cast<int64_t*>(data_offsets.data_ptr<int64_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(random_gather_copy_vector<PTYPE, MYTPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    random_gather_copy_vector<PTYPE, MYTPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr, 
        key_buffer_ptr, 
        value_data_ptr,
        value_buffer_ptr,
        src_metadata_ptr,
        dst_metadata_ptr,
        data_length,
        buffer_length,
        data_offsets_ptr,
        index_size,
        copy_start,
        copy_num);
}


// gather copy keys, values and from three src and concat into one buffer
void gather_copy_and_concat(
    torch::Tensor& key_data1,    // input, shape (groups, num_vectors1, dim)
    torch::Tensor& key_data2,    // input, shape (groups, num_vectors2, dim)
    torch::Tensor& key_data3,    // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& key_buffer,   // output, shape (groups, buffer_num_vectors, dim)
    torch::Tensor& value_data1,  // input, shape (groups, num_vectors1, dim)
    torch::Tensor& value_data2,  // input, shape (groups, num_vectors2, dim)
    torch::Tensor& value_data3,  // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& value_buffer, // output, shape (groups, buffer_num_vectors, dim)
    
    torch::Tensor& data_offsets2,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes2,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets2,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks2,        // input, shape (groups,)
    torch::Tensor& data_offsets3,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes3,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets3,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks3,        // input, shape (groups,)

    torch::Tensor& valid_lengths,       // output, shape (groups,)

    int groups, 
    int data_length1,           // num_vectors1
    int data_length2,           // num_vectors2
    int data_length3,           // num_units3
    int buffer_length,          // buffer_num_vectors
    int offset_length,
    torch::Tensor& copy_vectors1 // number of vectors need to copy from data1
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = (CHUNK_SIZE + 1 + offset_length * 3) * sizeof(int);

    PTYPE* key_data_ptr1;
    PTYPE* key_data_ptr2;
    PTYPE* key_data_ptr3;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr1;
    PTYPE* value_data_ptr2;
    PTYPE* value_data_ptr3;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<float>());
    key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<float>());
    key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<float>());
    value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<float>());
    value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data1.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::Half>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::Half>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::Half>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::Half>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::BFloat16>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::BFloat16>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::BFloat16>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::BFloat16>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif
    
    int* data_offsets_ptr2 = reinterpret_cast<int*>(data_offsets2.data_ptr<int32_t>());
    int* data_copy_sizes_ptr2 = reinterpret_cast<int*>(data_copy_sizes2.data_ptr<int32_t>());
    int* buffer_offsets_ptr2 = reinterpret_cast<int*>(buffer_offsets2.data_ptr<int32_t>());
    int* copy_chunks_ptr2 = reinterpret_cast<int*>(copy_chunks2.data_ptr<int32_t>());
    int* data_offsets_ptr3 = reinterpret_cast<int*>(data_offsets3.data_ptr<int32_t>());
    int* data_copy_sizes_ptr3 = reinterpret_cast<int*>(data_copy_sizes3.data_ptr<int32_t>());
    int* buffer_offsets_ptr3 = reinterpret_cast<int*>(buffer_offsets3.data_ptr<int32_t>());
    int* copy_chunks_ptr3 = reinterpret_cast<int*>(copy_chunks3.data_ptr<int32_t>());

    int* valid_lengths_ptr = reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>());
    int* copy_vectors1_ptr = reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(concat_gather_copy<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    concat_gather_copy<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr1, 
        key_data_ptr2,
        key_data_ptr3,
        key_buffer_ptr, 
        value_data_ptr1,
        value_data_ptr2,
        value_data_ptr3,
        value_buffer_ptr,
        data_length1*128,
        data_length2*128,
        data_length3*CHUNK_SIZE*128,
        buffer_length*128,
        copy_vectors1_ptr,
        offset_length,
        data_offsets_ptr2,
        data_copy_sizes_ptr2,
        buffer_offsets_ptr2,
        copy_chunks_ptr2,
        data_offsets_ptr3,
        data_copy_sizes_ptr3,
        buffer_offsets_ptr3,
        copy_chunks_ptr3,
        valid_lengths_ptr
    );
}


// gather copy from src and scatter to dst
void gather_copy_and_scatter(
    torch::Tensor& key_data,    // input, src values, shape (groups, num_vectors, dim)
    torch::Tensor& key_buffer,  // inout, dst values, shape (groups, buffer_num_units, unit_size, dim)
    torch::Tensor& value_data,  // input, src values, shape (groups, num_vectors, dim)
    torch::Tensor& value_buffer,// inout, dst values, shape (groups, buffer_num_units, unit_size, dim)

    torch::Tensor& s_offsets,   // input, src offsets (groups, offset_length)
    torch::Tensor& copy_sizes,  // input, copy vector number for each chunk (groups, offset_length)
    torch::Tensor& d_offsets,   // input, dst offsets (groups, offset_length)
    torch::Tensor& copy_chunks, // input, number of chunks need to copy (groups,) 

    int groups, 
    int data_length,            // num_vectors
    int buffer_length,          // buffer_num_units
    int offset_length,
    torch::Tensor& s_copy_start // start vector index for copy src
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups;
    const int maxSMBytes = (CHUNK_SIZE + 1 + offset_length * 3) * sizeof(int);

    PTYPE* key_data_ptr;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data.dtype() == torch::kFloat16) {      // fp16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif

    int* src_offsets_ptr = reinterpret_cast<int*>(s_offsets.data_ptr<int32_t>());
    int* copy_sizes_ptr = reinterpret_cast<int*>(copy_sizes.data_ptr<int32_t>());
    int* dst_offsets_ptr = reinterpret_cast<int*>(d_offsets.data_ptr<int32_t>());
    int* copy_chunks_ptr = reinterpret_cast<int*>(copy_chunks.data_ptr<int32_t>());

    int* s_copy_start_ptr = reinterpret_cast<int*>(s_copy_start.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gather_copy_scatter<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    gather_copy_scatter<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr, 
        key_buffer_ptr, 
        value_data_ptr,
        value_buffer_ptr,
        data_length*128, 
        buffer_length*CHUNK_SIZE*128,
        offset_length,
        s_copy_start_ptr,
        src_offsets_ptr,
        copy_sizes_ptr,
        dst_offsets_ptr,
        copy_chunks_ptr
    );
}


// reorganize keys, values into clusters
void reorganize_vectors(
    torch::Tensor& key_src,         // input, shape (groups, src_num_vectors, dim)
    torch::Tensor& key_dst,         // output, shape (groups, dst_num_vectors, dim)
    torch::Tensor& value_src,       // input, shape (groups, src_num_vectors, dim)
    torch::Tensor& value_dst,       // output, shape (groups, dst_num_vectors, dim)
    
    torch::Tensor& src_offsets,     // input, shape (groups, copy_cluster_num, max_cluster_size)
    torch::Tensor& dst_cumsum,      // input, shape (groups, n_centroids)

    int groups, 
    int copy_start                  // start cluster id to copy
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;

    int copy_cluster_num = src_offsets.size(-2);
    int max_cluster_size = src_offsets.size(-1);
    int n_centroids = dst_cumsum.size(-1);
    int src_length = key_src.size(-2);
    int dst_length = key_dst.size(-2);

    int split_size = (copy_cluster_num + SPLIT_FACTOR - 1) / SPLIT_FACTOR;
    const int maxSMBytes = (split_size*max_cluster_size + split_size + split_size) * sizeof(int);

    PTYPE* key_src_ptr;
    PTYPE* key_dst_ptr;
    PTYPE* value_src_ptr;
    PTYPE* value_dst_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<float>());
    key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<float>());
    value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<float>());
    value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<float>());
#else
    if (key_src.dtype() == torch::kFloat16) {     // fp16
        key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<at::Half>());
        key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<at::Half>());
        value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<at::Half>());
        value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<at::Half>());
    } else {    // bf16
        key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<at::BFloat16>());
        key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<at::BFloat16>());
        value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<at::BFloat16>());
        value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<at::BFloat16>());
    }
#endif
    
    int* src_offsets_ptr = reinterpret_cast<int*>(src_offsets.data_ptr<int>());
    int* dst_cumsum_ptr = reinterpret_cast<int*>(dst_cumsum.data_ptr<int>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gather_copy_append<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    gather_copy_append<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_src_ptr, 
        key_dst_ptr, 
        value_src_ptr,
        value_dst_ptr,
        src_length*128,
        dst_length*128,
        src_offsets_ptr,
        dst_cumsum_ptr,
        copy_cluster_num,
        max_cluster_size,
        n_centroids,
        copy_start);
}


// gather copy keys, values and from two src (streaming copy + cluster copy) and concat into one buffer
void gather_copy_cluster_and_concat_fuse(
    torch::Tensor& key_data1,    // input, shape (groups, num_vectors1, dim)
    torch::Tensor& key_data2,    // input, shape (groups, num_vectors2, dim)
    torch::Tensor& key_buffer,   // output, shape (groups, buffer_num_vectors, dim)
    torch::Tensor& value_data1,  // input, shape (groups, num_vectors1, dim)
    torch::Tensor& value_data2,  // input, shape (groups, num_vectors2, dim)
    torch::Tensor& value_buffer, // output, shape (groups, buffer_num_vectors, dim)
    
    torch::Tensor& cluster_cumsum,      // input, shape (groups, n_centroids)
    torch::Tensor& select_cluster_ids,  // input, shape (groups, index_size)
    torch::Tensor& valid_lengths,       // output, shape (groups,)

    int groups, 
    int data_length1,               // num_vectors1
    int data_length2,               // num_vectors2
    int buffer_length,              // buffer_num_vectors

    int nprobe_for_mem_alloc,       // used to compute shared memory size
    torch::Tensor& nprobe_tensor,   // input, zero-dim tensor, select cluster number
    torch::Tensor& copy_vectors1    // input, zero-dim tensor, number of vectors need to copy from data1
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = nprobe_for_mem_alloc * 3 * sizeof(int);

    int n_centroids = cluster_cumsum.size(1);
    int index_size = select_cluster_ids.size(1);

    PTYPE* key_data_ptr1;
    PTYPE* key_data_ptr2;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr1;
    PTYPE* value_data_ptr2;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<float>());
    key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<float>());
    value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data1.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::Half>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::Half>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::BFloat16>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::BFloat16>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif
    
    int* cluster_cumsum_ptr = reinterpret_cast<int*>(cluster_cumsum.data_ptr<int32_t>());
    int64_t* select_cluster_ids_ptr = reinterpret_cast<int64_t*>(select_cluster_ids.data_ptr<int64_t>());
    int* valid_lengths_ptr = reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>());
    int* copy_vectors1_ptr = reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>());
    int* nprobe_tensor_ptr = reinterpret_cast<int*>(nprobe_tensor.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(concat_gather_copy_clusters_fuse<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    concat_gather_copy_clusters_fuse<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr1, 
        key_data_ptr2,
        key_buffer_ptr, 
        value_data_ptr1,
        value_data_ptr2,
        value_buffer_ptr,
        data_length1*128,
        data_length2*128,
        buffer_length*128,
        cluster_cumsum_ptr,
        select_cluster_ids_ptr,
        n_centroids,
        index_size,
        buffer_length,
        nprobe_tensor_ptr,
        copy_vectors1_ptr,
        valid_lengths_ptr);
}



namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_copy_vectors", &gather_copy_vectors, "Gather copy keys, values and metadata in vector-level (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("src_metadata"), py::arg("dst_metadata"), py::arg("data_offsets"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("index_size"), py::arg("copy_start"), py::arg("copy_num")),

    m.def("gather_copy_and_concat", &gather_copy_and_concat, "Gather copy and concat from different sources (CUDA)",
    py::arg("key_data1"), py::arg("key_data2"), py::arg("key_data3"), py::arg("key_buffer"),
    py::arg("value_data1"), py::arg("value_data2"), py::arg("value_data3"), py::arg("value_buffer"),
    py::arg("data_offsets2"), py::arg("data_copy_sizes2"), py::arg("buffer_offsets2"), py::arg("copy_chunks2"),
    py::arg("data_offsets3"), py::arg("data_copy_sizes3"), py::arg("buffer_offsets3"), py::arg("copy_chunks3"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("data_length1"), py::arg("data_length2"), py::arg("data_length3"), 
    py::arg("buffer_length"), py::arg("offset_length"), py::arg("copy_vectors1")),

    m.def("gather_copy_and_scatter", &gather_copy_and_scatter, "Gather copy from src and scatter to dst (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("s_offsets"), py::arg("copy_sizes"), py::arg("d_offsets"), py::arg("copy_chunks"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("offset_length"), py::arg("s_copy_start")),

    m.def("reorganize_vectors", &reorganize_vectors, "Reorganize keys, values into clusters (CUDA)",
    py::arg("key_src"), py::arg("key_dst"), py::arg("value_src"), py::arg("value_dst"),
    py::arg("src_offsets"), py::arg("dst_cumsum"),
    py::arg("groups"), py::arg("copy_start")),

    m.def("gather_copy_cluster_and_concat_fuse", &gather_copy_cluster_and_concat_fuse, "Gather copy clusters and concat with fuse (CUDA)",
    py::arg("key_data1"), py::arg("key_data2"), py::arg("key_buffer"),
    py::arg("value_data1"), py::arg("value_data2"), py::arg("value_buffer"),
    py::arg("cluster_cumsum"), py::arg("select_cluster_ids"), py::arg("valid_lengths"),
    py::arg("groups"), py::arg("data_length1"), py::arg("data_length2"), py::arg("buffer_length"),
    py::arg("nprobe_for_mem_alloc"), py::arg("nprobe_tensor"), py::arg("copy_vectors1"));
}