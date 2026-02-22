import os
import sys
import json
import time
import torch
import argparse
import random
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_len", type=int, default=4096, help="Generation length")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"], help="huggingface model name")
    parser.add_argument("--chunk_size", type=int, default=-1, help="Chunk size for chunked prefill")
    parser.add_argument("--context_len", type=int, default=120000, help="Input context length")
    parser.add_argument("--task_name", type=str, default="NIAH", choices=["NIAH", "AIME"],
                        help="Test task name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    print(args)

    model_name = args.model_name
    batch_size = args.batch_size
    dtype = torch.bfloat16
    device = "cuda:0"
    task_name = args.task_name
    chunk_size = args.chunk_size

    # load input data
    TEST_DIR = os.path.join(PROJECT_ROOT, "throughput_eval/test_data")
    if task_name == "NIAH":
        TEST_FILE = os.path.join(TEST_DIR, f"NIAH_{args.context_len}.json")
        data = json.load(open(TEST_FILE))[0]
        prompt = data['input']
        groundtruth = data['answer']
    else:
        TEST_FILE = os.path.join(TEST_DIR, f"{task_name}.json")
        data = json.load(open(TEST_FILE))
        prompt = data['input']
        groundtruth = data['outputs']
    print(f"Loaded test data from {TEST_FILE}")
    prompts = [prompt for _ in range(batch_size)]

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    # inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    # print(f"input shape: {inputs.input_ids.shape}, output length: {args.gen_len}")

    if chunk_size > 0:
        llm = LLM(model=model_name, dtype=dtype, device=device, max_model_len=130000, gpu_memory_utilization=0.96, 
                  enable_chunked_prefill=True, max_num_batched_tokens=chunk_size, max_num_seqs=batch_size)
    else:
        llm = LLM(model=model_name, dtype=dtype, device=device, max_model_len=130000, gpu_memory_utilization=0.96, 
                  enable_chunked_prefill=False, max_num_seqs=batch_size)
    sampling_params = SamplingParams(n=1, max_tokens=args.gen_len, ignore_eos=True, temperature=0.0)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    end = time.time()
    print(f"End2End time: {end - start} seconds")

    total_execution_time = 0
    for output in outputs:
        print(f"arrival time: {output.metrics.arrival_time}, first_token_time: {output.metrics.first_token_time}, finished_time: {output.metrics.finished_time}")
        total_execution_time += output.metrics.finished_time - output.metrics.arrival_time
    avg_latency = total_execution_time / batch_size
    print(f"Avg. E2E Latency {avg_latency} s/req")

    # for output in outputs:
    #     generated_text = output.outputs[0].text
    #     print(f"Answer: {groundtruth}\nGenerated text: {[generated_text]}")
    