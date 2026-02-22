import os
import sys
import json
import torch
import argparse
import random
import numpy as np
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model_hub import load_model, load_tokenizer, add_model_args
from config import generate_config, add_config_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Total Batch size")
    parser.add_argument("--prefill_bsz", type=int, default=1, help="Prefilling batch size")
    parser.add_argument("--prefill_method", type=str, default="full", choices=["full", "xattn", "minfer"],
                        help="Prefilling method")
    parser.add_argument("--context_len", type=int, default=120000, help="Input context length")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--task_name", type=str, default="NIAH", choices=["NIAH", "fwe", "vt", "qa1", "AIME"],
                        help="Test task name")
    parser = add_model_args(parser)
    parser = add_config_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    print(args)

    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.bfloat16
    device = args.device
    task_name = args.task_name

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
    prompts = [prompt for _ in range(batch_size)]

    tokenizer = load_tokenizer(model_name)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask
    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}, Gen length: {gen_len}", 'yellow'))

    attn_config = generate_config(model_name, input_len, attn_type, 
                                  float(args.retrieval_budget), float(args.estimation_budget), float(args.cache_ratio),
                                  args.use_cuda_graph, args.gpu_only)
    llm = load_model(model_name, max_len, dtype, device)

    out = llm.generate(
        attention_type=attn_type,
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks,
        max_new_length=gen_len, 
        attn_config=attn_config,
        do_sample=False, 
        ignore_eos=True,
        prefill_bsz=args.prefill_bsz,
        prefill_method=args.prefill_method
    )
    
    if gen_len <= 100:
        result = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(groundtruth)
        print(result)