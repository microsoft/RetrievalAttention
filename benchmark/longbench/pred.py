import os
import sys
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, add_model_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import add_config_args, generate_config


model2path = json.load(open("config/model2path.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # overwrite model_name argument
    parser.add_argument('--model', type=str, default="llama-3-8b-1048k", 
                        choices=["llama-3-8b-1048k", "qwen2.5-7b", "llama-3.1-8b", "qwen2.5-72b"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--task', type=str, required=True, help="task name. work when --e is false")
    parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")

    parser = add_model_args(parser)
    parser = add_config_args(parser)

    return parser.parse_args(args)


def get_pred(llm, data, max_new_tokens, prompt_format, model_name, out_path, args):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        attn_config = generate_config(
            model2path[model_name], 
            input_ids.shape[1], 
            attn_type,
            retrieval_budget=args.retrieval_budget,
            estimation_budget=args.estimation_budget,
        )

        out = llm.generate(
            attention_type=attn_type,
            inputs_ids=input_ids.to(llm.layers[0].device),
            attention_masks=attention_masks.to(llm.layers[0].device),
            max_new_length=max_new_tokens, 
            attn_config=attn_config,
            do_sample=False,
            ignore_eos=True
        )

        output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
        pred = output[0]
        
        torch.cuda.empty_cache()
        
        print("Chunked generation:", pred[:50])

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred, 
                    "answers": json_obj["answers"], 
                    "all_classes": json_obj["all_classes"], 
                    "length": json_obj["length"]
                }, 
                f, 
                ensure_ascii=False
            )
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, max_len, dtype, device):
    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    
    return llm



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    num_examples = args.num_examples
    attn_type = args.attn_type
    model_name = args.model
    device = args.device
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16

    max_length = model2maxlen[model_name]
    model_path = model2path[model_name]

    llm = load_model(model_path, max_length, dtype, device)

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [args.task]
    
    # predict on each dataset
    if not os.path.exists("results/pred"):
        os.makedirs("results/pred")
    if not os.path.exists("results/pred_e"):
        os.makedirs("results/pred_e")

    for dataset in datasets:
        print(f"Predict {dataset}")
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)

            prefix = f"results/pred_e/{model_name}/{attn_type}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)

            prefix = f"results/pred/{model_name}/{attn_type}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_all = data_all[:num_examples] if num_examples > 0 else data_all

        get_pred(
            llm,
            data_all,
            max_new_tokens,
            prompt_format,
            model_name,
            out_path,
            args,
        )