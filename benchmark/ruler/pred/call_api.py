# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import torch
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
import traceback
from utils import load_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, add_model_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import add_config_args, generate_config


SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class HuggingFaceModel:
    def __init__(self, model_name, max_len, max_new_len, attn_type, dtype, 
                 device, retrieval_budget, estimation_budget, synthetic_len,
                 args) -> None:
        if 'Llama' in model_name:
            llm = LlamaModel(model_name,
                max_length=max_len+max_new_len,
                dtype=dtype,
                device_map=device)
        elif 'Qwen' in model_name:
            llm = QwenModel(model_name,
                max_length=max_len+max_new_len,
                dtype=dtype,
                device_map=device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.llm = llm
        self.max_new_len = max_new_len
        self.attn_type = attn_type
        self.dtype = dtype

        self.model_name = model_name
        self.retrieval_budget = retrieval_budget
        self.estimation_budget = estimation_budget
        self.synthetic_len = synthetic_len
        self.args = args

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        generated_text = get_pred(
            self.llm,
            input_text=prompt, 
            max_new_tokens=self.max_new_len,
            attn_type=self.attn_type,
            dtype=self.dtype,
            model_name=self.model_name,
            retrieval_budget=self.retrieval_budget,
            estimation_budget=self.estimation_budget,
            synthetic_len=self.synthetic_len,
            args=self.args,
        )
        return {'text': [generated_text]}


def get_llm(model_name, max_len, max_new_len, attn_type, dtype, 
            device, retrieval_budget, estimation_budget, synthetic_len,
            args):
    if args.server_type == 'hf':
        llm = HuggingFaceModel(
            model_name=model_name,
            max_len=max_len,
            max_new_len=max_new_len,
            attn_type=attn_type,
            dtype=dtype,
            device=device,
            retrieval_budget=retrieval_budget,
            estimation_budget=estimation_budget,
            synthetic_len=synthetic_len,
            args=args,
        ) 
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def get_pred(
    llm,
    input_text: str,
    max_new_tokens: int,
    attn_type: str,
    dtype: torch.dtype,
    model_name: str,
    retrieval_budget: float,
    estimation_budget: float,
    synthetic_len: int,
    args,
) -> str:
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    inputs = llm.tokenizer([input_text], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    attn_config = generate_config(
        model_name, 
        synthetic_len, 
        attn_type,
        retrieval_budget=retrieval_budget,
        estimation_budget=estimation_budget,
    )

    out = llm.generate(
        attention_type=attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=max_new_tokens, 
        attn_config=attn_config,
        do_sample=False,
        ignore_eos=True,
        prefill_method=args.prefill_method,
    )

    output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    print("Chunked generation:", output[0])
    return output[0]


def get_output(llm, outputs_parallel, idx, index, input, outputs, others, truncation, length):
    while True:
        try:
            pred = llm(prompt=input)
            break
        except Exception as e:
            traceback.print_exc()

    if len(pred['text']) > 0:
        outputs_parallel[idx] = {
            'index': index,
            'pred': pred['text'][0],
            'input': input,
            'outputs': outputs,
            'others': others,
            'truncation': truncation,
            'length': length,
        }


def main(args):
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in load_data(pred_file)]
        data = [sample for sample in load_data(task_file) if sample['index'] not in pred_index]
    else:
        data = load_data(task_file)

    # Load api
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16

    assert args.max_len >= args.synthetic_len, \
        f"max model length {args.max_len} should be no smaller than context length {args.synthetic_len}"
    
    llm = get_llm(args.model_name, args.max_len, config['tokens_to_generate'], 
                  args.attn_type, dtype, args.device, 
                  retrieval_budget=args.retrieval_budget, estimation_budget=args.estimation_budget, 
                  synthetic_len=args.synthetic_len, args=args)
    
    threads = []
    outputs_parallel = [{} for _ in range(len(data))]
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    llm=llm,
                    outputs_parallel=outputs_parallel,
                    idx=idx,
                    index=data_point['index'],
                    input=data_point['input'],
                    outputs=data_point['outputs'],
                    others=data_point.get('others', {}),
                    truncation=data_point.get('truncation', -1),
                    length=data_point.get('length', -1),
                ),
            )
            thread.start()
            threads.append(thread)
            if len(threads) == args.threads:
                for thread in threads:
                    thread.join()
                threads = []
                for computed_idx in range(idx - args.threads + 1, idx + 1):
                    if len(outputs_parallel[computed_idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')

        # collecting the final batch
        if len(data) > 0:
            for thread in threads:
                thread.join()
            for computed_idx in range(idx - len(threads) + 1, idx + 1):
                if len(outputs_parallel[computed_idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
    parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
    parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
    parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

    # Server
    parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default='127.0.0.1')
    parser.add_argument("--server_port", type=str, default='5000')
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)

    # Inference
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--synthetic_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prefill_method", type=str, default="full", choices=["full", "xattn", "minfer"], 
                        help="Prefilling method")

    parser = add_config_args(parser)
    parser = add_model_args(parser)

    args = parser.parse_args()
    print(args)

    if args.server_type == 'hf' or args.server_type == 'gemini':
        args.threads = 1
    
    seed_everything(2025)
    main(args)