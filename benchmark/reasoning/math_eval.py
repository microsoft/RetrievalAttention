import os
import sys
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import torch

from evaluate_utils import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import generate_completions, load_lm_and_tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import add_config_args



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24", type=str, choices=["aime24", "gpqa"])
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", type=str, 
                        choices=["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"])
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--prompt_type", default="orz", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--max_tokens_per_call", default=32768, type=int, help="Max new tokens to generate")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--adapt_few_shot", action="store_true", 
                        help="Few shot for multiple-choice questions, zero shot for others.")

    # new args
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=65536, help="max length for model")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--do_sample", action="store_true")
    parser = add_config_args(parser)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy sampling (vllm)
    
    return args


def prepare_data(data_name, args):
    """
    Return all example to process, processed, and out_file path
    Return:
    - examples: {}
    - processed: processed examples
    - out_file: output file path
    """
    examples = load_data(data_name, args.split, args.data_dir, args)

    # select start and end
    print(f"{len(examples)} examples loaded.")
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    
    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    print(f"{len(examples)} examples to eval. idx range: {examples[0]['idx']} to {examples[-1]['idx']}")

    # # shuffle
    # if args.shuffle:
    #     random.seed(datetime.now().timestamp())
    #     random.shuffle(examples)

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_{args.attn_type}_budget{args.retrieval_budget}_es{args.estimation_budget}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_{dt_string}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
    llm, tokenizer = load_lm_and_tokenizer(
        model_path=args.model_name_or_path,
        max_len=args.max_length,
        dtype=dtype,
        device="auto",
    )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to results
    data_list.append("avg")
    results.append(
        {
            "avg_acc": sum([result["avg_acc"] for result in results]) / len(results),
            "pass@1": sum([result["pass@1"] for result in results]) / len(results),
        }
    )

    # print results
    print(f"\nPass@{args.n_sampling}:")
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['pass@1']:.1f}".ljust(pad, " ") for result in results]))
    
    # print(f"\nAccuracy:")
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['avg_acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, ", eval samples:", len(examples))
    if len(examples) > 0:
        print(f"data example: {examples[0]}\n")

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # prepare all samples
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    # start inference
    # start_time = time.time()
    for epoch in range(max_func_call):
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            print(f"all prompts are processed, break.")
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]

        outputs = generate_completions(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_tokens_per_call,
            batch_size=args.batch_size,
            stop_id_sequences=None,
            args=args,
        )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if "boxed" not in output and output.endswith("```"):
                print(f"query {i} output with code and no boxed")
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                # print(f"query {i} finished")
                end_prompts.append((i, query))

        # execute the remain prompts
        print(f"\n==================== execute the remain prompts ====================")
        print(f"num of remain prompts: {len(remain_prompts)} == {len(remain_codes)}")
        remain_results = executor.batch_apply(remain_codes)

        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in [llm.tokenizer.eos_token]:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    # time_use = time.time() - start_time

    print(f"\n-------------\nall results:\n{results}\n-------------\n")

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # for i in range(len(all_samples)):
    #     print(f"preds: {i}: {all_samples[i]['pred']}\n")

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # result_json["time_use_in_second"] = time_use
    # result_json["time_use_in_minite"] = (
    #     f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    # )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)