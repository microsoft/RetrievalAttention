# RetroInfer

[RetroInfer](https://arxiv.org/pdf/2505.02922) is a novel system that **rethinks the KV cache as vector storage** within a GPU–CPU co-execution setup to accelerate long-context LLM inference. It exploits the inherent sparsity of the attention mechanism and introduces an **A**ttention-a**W**are **VE**ctor index (*wave index*) that enables efficient and accurate retrieval of critical tokens from the KV cache. Complementing this is the *wave buffer*, which coordinates KV cache placement and overlaps computation and data transfer across GPU and CPU to sustain high throughput.

## Getting Started

### Environment Setup
The required dependency packages rely on `CUDA 12.4`, you can use the docker image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` if your system does not have CUDA 12.4 installed.

The code was tested with `Python 3.10.16`, we recommend using `conda` to mange your Python environments:
```bash
# firstly install miniconda if you don't have it, then create a new conda environment:
conda create -n retroinfer python=3.10 -y
conda activate retroinfer 

# install conda packages
conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y

# may need to downgrade pip to <=25.0 to solve `DEPRECATION warning` when using `pip install .` to install kernels
python -m pip install pip==25.0

# install python packages
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/
pip install git+https://github.com/Starmys/flash-attention.git@weighted
```

### Install RetroInfer Kernels
```bash
cd library/
git clone https://github.com/NVIDIA/cutlass.git
cd retroinfer && pip install . && cd ../../
```

### Simple Test
We provide a simple demo to verify that the environment is set up correctly. The demo runs on four different contexts from [RULER](https://github.com/NVIDIA/RULER), each containing approximately 120,000 tokens. You can run the demo using the following command:
```bash
python -u simple_test.py --batch_size 4
```
Running this demo requires about 25GB GPU memory and 67GB CPU memory. If you encounter out-of-memory errors, consider reducing the batch size.

You can also customize the input contexts by providing a `json` file in the following format:
```
[
    {"input": str, "outputs": str}, 
    {"input": str, "outputs": str},
    ...
]
``` 
Then, pass the file path using the `--data_path` argument:
```bash
python -u simple_test.py --data_path <your_json_file_path>
```

## Run Benchmark

You may need to set `CUDA_VISIBLE_DEVICES` before running the benchmark since our code will automatically split models into all visable GPUs. For example, when evaluating with A100 80GB, 7B/8B models only need one GPU card while 72B models need at least 3 GPU cards. 

### [RULER](https://github.com/NVIDIA/RULER)
To evaluate the model accuracy on the RULER benchmark, you need firstly download the benchmark datasets:
```bash
cd benchmark/ruler
cd data/synthetic/json/ && python -u download_paulgraham_essay.py && bash download_qa_dataset.sh && cd ../../../
```
Then, you can run [ruler_run.sh](benchmark/ruler/ruler_run.sh) to evaluate.
For example, you can evaluate `RetroInfer` on RULER variable tracing task `vt` with the context length of `128K` using the following command:
```bash
bash ruler_run.sh llama-3-8b-1048k synthetic RetroInfer 131072 vt bf16 0.018 0.232
```
The input parameters of the evaluation script are, in order:
- `model name`: supported models include `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b` and `qwen2.5-72b`;
- `benchmark name`: set to `synthetic`;
- `attention type`: `RetroInfer` or `Full_Flash_Attn`;
- `input context length`: the input context length;
- `evaluate task name`: supported tasks include `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1` and `qa_2`;
- `model data type`: supported data types include `bf16` and `fp16`;
- `retrieval budget ratio`: the ratio of the number of tokens to be retrieved from the KV cache to the total number of tokens in the input context;
- `attention estimate ratio`: the ratio of the number of clusters to be estimated in the attention mechanism to the total number of clusters.

### [LongBench](https://github.com/THUDM/LongBench)
You can use the following command to evaluate the model accuracy of `RetroInfer` on the LongBench:
```bash
cd benchmark/LongBench
bash longbench_run.sh llama-3-8b-1048k RetroInfer 0.018 0.232 bf16
```
The input parameters of the evaluation script are, in order:
- `model name`: supported models include `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b` and `qwen2.5-72b`;
- `attention type`: `RetroInfer` or `Full_Flash_Attn`;
- `retrieval budget ratio`: the ratio of the number of tokens to be retrieved from the KV cache to the total number of tokens in the input context;
- `attention estimate ratio`: the ratio of the number of clusters to be estimated in the attention mechanism to the total number of clusters;
- `model data type`: supported data types include `bf16` and `fp16`.

## Reproduce Throughput Results
We provide scripts to reproduce the throughput results reported in the paper. These experiments were conducted on an [Azure virtual machine](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndma100v4-series?tabs=sizebasic) featuring 4 NUMA nodes. Each NUMA node is equipped with 24 CPU cores, 475 GB of CPU memory, and two 80GB A100 GPUs.
```bash
# Firstly, install the numactl package
sudo apt install numactl -y

# run scripts
cd throughput_eval
bash run.sh
```

## Reference

If you find this project helpful, please cite our paper:
```bibtex
@misc{chen2025retroinfervectorstorageapproachscalable,
    title={RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference}, 
    author={Yaoqi Chen and Jinkai Zhang and Baotong Lu and Qianxi Zhang and Chengruidong Zhang and Jingjia Luo and Di Liu and Huiqiang Jiang and Qi Chen and Jing Liu and Bailu Ding and Xiao Yan and Jiawei Jiang and Chen Chen and Mingxing Zhang and Yuqing Yang and Fan Yang and Mao Yang},
    year={2025},
    eprint={2505.02922},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2505.02922}, 
}

@misc{liu2024retrievalattentionacceleratinglongcontextllm,
      title={RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval}, 
      author={Di Liu and Meng Chen and Baotong Lu and Huiqiang Jiang and Zhenhua Han and Qianxi Zhang and Qi Chen and Chengruidong Zhang and Bailu Ding and Kai Zhang and Chen Chen and Fan Yang and Yuqing Yang and Lili Qiu},
      year={2024},
      eprint={2409.10516},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.10516}, 
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
