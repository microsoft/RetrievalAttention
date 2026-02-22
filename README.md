# RetroInfer

[RetroInfer](https://arxiv.org/pdf/2505.02922) is a novel system that rethinks the KV cache as vector storage within a GPU–CPU co-execution setup to accelerate long-context LLM inference. It exploits the inherent sparsity of the attention mechanism and introduces an **A**ttention-a**W**are **VE**ctor index (*wave index*) that enables efficient and accurate retrieval of critical tokens from the KV cache. Complementing this is the *wave buffer*, which coordinates KV cache placement and overlaps computation and data transfer across GPU and CPU to sustain high throughput. Key ideas behind RetroInfer include:
- Attention is split into three zones: `steady`, `retrieval`, and `estimation`, allowing us to handle dynamic sparsity with *accuracy-bounded attention estimation*.
- By leveraging coarse-grained spatial locality in attention, we design a lightweight *segmented clustering* algorithm for low-overhead index construction and updates.
- Highly-optimized CUDA kernels to support fast GPU–CPU data movement and sustain high throughput.

<div align="center">
  <img src="asserts/RetroInfer.png" width="500"/>
  <p><em>RetroInfer Architecture.</em></p>
</div>

## :zap: Getting Started

### Environment Setup
The required dependency packages rely on `CUDA 12.4`, you can use the docker image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` if your system does not have `CUDA 12.4` installed.

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

### Install Kernels
```bash
cd library/
git clone https://github.com/NVIDIA/cutlass.git
cd retroinfer && pip install . && cd ..

# If you want to use MInference, install the following package:
pip install minference==0.1.6.0

# If you want to use XAttention, install the following kernels:
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git 
cd Block-Sparse-Attention && git checkout 0e2478b0a4d9858cf0910f78a8aaf4fba751de69 && export MAX_JOBS=8 && python setup.py install && cd ..

# go back to root directory
cd ..
```

### Simple Test
We provide a simple demo to verify that the environment is set up correctly. The demo runs on four different contexts from [RULER](https://github.com/NVIDIA/RULER), each containing approximately 120,000 tokens. You can run the demo using the following command:
```bash
python -u simple_test.py --batch_size 4
```
It will execute RetroInfer on [Llama-3-8B-1048K](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) model. Running this demo requires about 35GB GPU memory and 70GB CPU memory. If you encounter out-of-memory errors, consider reducing the batch size.

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

There are several options you can set to customize the execution:
- Set `--gpu_only` to run GPU-only version of RetroInfer, which will keep all KV cache on GPU memory. 
- Set `--use_cuda_graph` to enable CUDA graphs, which can reduce the overhead of kernel launches and improve throughput.
- Set `--do_sample` to enable sampling during generation. 
- Use `--prefill_method` to specify the prefilling methods, currently support `full` (Full attention, default), `xattn` ([XAttention](https://arxiv.org/pdf/2503.16428)) and `minfer` ([MInference](https://arxiv.org/pdf/2407.02490)).

### API
We provide a simple API to use RetroInfer. Here is an example of how to use it:
```python
from model_hub import load_model, load_tokenizer
from config import generate_config

# load tokenizer and model
tokenizer = load_tokenizer(model_name)
llm = load_model(model_name, max_seq_len, dtype, device, tokenizer)

# load RetroInfer config
attn_config = generate_config(
    model_name, input_seq_len, "RetroInfer", 
    retrieval_budget, estimation_budget, cache_ratio,
    use_cuda_graph=False, gpu_only=False
)

# generate outputs
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_masks = inputs.attention_mask
out = llm.generate(
    attention_type=attn_type,
    inputs_ids=input_ids.to(llm.layers[0].device),
    attention_masks=attention_masks,
    max_new_length=gen_len, 
    attn_config=attn_config
)
result = tokenizer.batch_decode(out, skip_special_tokens=True)
```

## :dart: Run Benchmark

> [!IMPORTANT]
> You may need to set `CUDA_VISIBLE_DEVICES` before running the benchmark since our code will automatically split models into all visable GPUs. For example, when evaluating with A100 80GB, 7B/8B models only need one GPU card while 72B models need at least 3 GPU cards. 

> [!NOTE]
> Benchmark results may slightly differ from the numbers reported in the paper due to hardware-related randomness.

### [RULER](https://github.com/NVIDIA/RULER)
To evaluate the model accuracy on the RULER benchmark, you need firstly download the benchmark datasets:
```bash
cd benchmark/ruler
cd data/synthetic/json/ && python -u download_paulgraham_essay.py && bash download_qa_dataset.sh && cd ../../../
```
Then, you can run [ruler_run.sh](benchmark/ruler/ruler_run.sh) to evaluate. For example, you can evaluate RetroInfer on RULER variable tracing task (`vt`) at 128K context length using the following command:
```bash
bash ruler_run.sh llama-3-8b-1048k full RetroInfer 131072 vt bf16 0.018 0.232
```
The input parameters of the evaluation script are, in order:
- `model name`: supported models include `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b` and `qwen2.5-72b`;
- `prefill method`: supported prefilling methods include `full`, `xattn` and `minfer`;
- `attention type`: `RetroInfer` or `Full_Flash_Attn`;
- `input context length`: the input context length;
- `evaluate task name`: supported tasks include `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1` and `qa_2`;
- `model data type`: supported data types include `bf16` and `fp16`;
- `retrieval budget ratio`: the ratio of the number of tokens to be retrieved from the KV cache to the total number of tokens in the input context;
- `attention estimate ratio`: the ratio of the number of clusters to be estimated in the attention mechanism to the total number of clusters.

### [LongBench](https://github.com/THUDM/LongBench)
You can use the following command to evaluate the model accuracy of RetroInfer on the Longbench benchmark:
```bash
cd benchmark/longbench
bash longbench_run.sh llama-3-8b-1048k RetroInfer 0.018 0.232 bf16 SQA
```
The input parameters of the evaluation script are, in order:
- `model name`: supported models include `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b` and `qwen2.5-72b`;
- `attention type`: `RetroInfer` or `Full_Flash_Attn`;
- `retrieval budget ratio`: the ratio of the number of tokens to be retrieved from the KV cache to the total number of tokens in the input context;
- `attention estimate ratio`: the ratio of the number of clusters to be estimated in the attention mechanism to the total number of clusters;
- `model data type`: supported data types include `bf16` and `fp16`;
- `sub categories`: supported categories include `SQA` (single-document qa), `MQA` (multi-documents qa), `SUM` (summarization), `FSL` (few-shot learning), `ST` (synthetic tasks) and `CC` (code completion).

### [Reasoning Benchmark](https://github.com/QwenLM/Qwen2.5-Math)
To evaluate the model accuracy on the long reasoning tasks, you need firstly install the dependencies:
```bash
cd benchmark/reasoning/latex2sympy && pip install -e . && cd ..
pip install -r requirements.txt 
```

Then you can use the following command to evaluate the model accuracy of RetroInfer on reasoning tasks with the default setting: 
```bash
bash eval.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B RetroInfer aime24 0 -1
```
The input parameters of the evaluation script are, in order:
- `model_name_or_path`: support `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`;
- `data_names`: support `aime24` and `gpqa`;
- `attention type`: `RetroInfer` or `Full_Flash_Attn`;
- `eval start index`: the start index of the evaluation samples;
- `eval number`: the number of evaluation samples. If set to `-1`, it means evaluating all samples.

Other key parameters of the corresponding python file [`math_eval.py`](./benchmark/reasoning/math_eval.py) are as follow:
- sampling parameters: `temperature`, `top_p`, `top_k` and `do_sample`. Default to `0.6`, `0.95`, `20`, `True` respectively;
- num of sampling: `n_sampling`. When this parameter is set to k, the system performs k independent sampling runs and evaluates the corresponding pass@k result.

## :bar_chart: Reproduce Throughput Results
We provide scripts to reproduce the throughput results reported in the paper. These experiments were conducted on an [Azure virtual machine](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndma100v4-series?tabs=sizebasic) featuring 4 NUMA nodes. Each NUMA node is equipped with 24 CPU cores, 475 GB of CPU memory, and two 80GB A100 GPUs.
```bash
# Firstly, install the numactl package
sudo apt install numactl -y

# run scripts
cd throughput_eval
bash run.sh
```

## :clipboard: Add New Sparsity Methods
This repository provides a flexible inference framework that allows users to easily integrate new sparsity-based attention methods. To add a new sparsity method, you can follow these steps: 
1. Add your KV cache management logic in `cache_hub/` directory.
2. Add your attention computation logic in `attn_hub/` directory.
3. Update [config.py](./config/config.py) to include configuration options for your new method.
4. Update `init_kv_cache()`, `decode_attention()` and `parameter_move()` functions in [llama.py](./model_hub/llama.py) and [qwen.py](./model_hub/qwen.py) to incorporate your new method.
5. Now you can try your new sparsity method by specifying it in the `--attn_type` argument when running the scripts.

## :bulb: Reference
If you find this project helpful, please cite our papers:
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