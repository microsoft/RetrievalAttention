"""
https://github.com/allenai/open-instruct
"""
import os
import sys
import torch
import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import generate_config

from parser import extract_answer


@torch.no_grad()
def generate_completions(llm, tokenizer, prompts, max_new_tokens,
                         batch_size=1, stop_id_sequences=None, 
                         add_special_tokens=True, disable_tqdm=False, 
                         args=None, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        print(f"\n--------------------- Test Data {i} ---------------------")
        print(batch_prompts)

        inputs = llm.tokenizer(batch_prompts, return_tensors="pt", padding=True, 
                               add_special_tokens=add_special_tokens)
        batch_input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        attn_config = generate_config(
            args.model_name_or_path, 
            batch_input_ids.shape[1], 
            args.attn_type,
            retrieval_budget=args.retrieval_budget,
            estimation_budget=args.estimation_budget,
        )
        attn_config["RetroInfer"]["buffer_cluster_num"] = 200

        batch_outputs = llm.generate(
            attention_type=args.attn_type,
            inputs_ids = batch_input_ids.to(llm.layers[0].device),
            attention_masks = attention_masks.to(llm.layers[0].device),
            max_new_length=max_new_tokens, 
            attn_config=attn_config,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            ignore_eos=False
        )

        torch.cuda.empty_cache()

        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        
        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        # remove the remain stop sequence from the output.
        stop_id_sequences = [tokenizer.eos_token] if stop_id_sequences is None else stop_id_sequences
        for idx, prediction in enumerate(batch_generations):
            for stop_sequence in stop_id_sequences:
                batch_generations[idx] = prediction.split(stop_sequence)[0]

        generations += batch_generations

        temp_batch_answer = []
        for answer in batch_generations:
            # print(f"answer: \n{answer}\n")
            pred = extract_answer(answer, args.data_names)
            if len(pred) > 50:
                pred = pred[-50:]
            temp_batch_answer.append(pred)
            # print(f"pred: \n{pred}\n")

        print(f"preds for test data {i}: \n{temp_batch_answer}")

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, \
           f"number of generations({len(generations)}) should be equal to number of prompts({len(prompts)}) * num_return_sequences({num_return_sequences})"
    return generations



def load_lm_and_tokenizer(model_path, max_len, dtype, device):
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
    
    return llm, llm.tokenizer
