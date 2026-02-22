import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

sub_categories = {
    "Single-Document QA": ["qasper", "multifieldqa_en", "narrativeqa"],
    "Multi-Document QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot learning": ["trec", "lsht", "samsum", "triviaqa"],
    "Synthetic tasks": ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],
    "Code Completion": ["repobench-p", "lcc"]
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, 
                        choices=["llama-3-8b-1048k", "qwen2.5-7b", "llama-3.1-8b", "qwen2.5-72b"])
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",
                        choices=["Full_Flash_Attn", "RetroInfer"],
                        help="Attention method")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    attn_type = args.attn_type
    model_name = args.model # not hf model path

    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]

    if args.e:
        path = f"results/pred_e/{args.model}/{attn_type}/"
    else:
        path = f"results/pred/{args.model}/{attn_type}/"
    
    scores = dict()
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    
    print(f"\n\nAverage scores of sub-categories:")
    for category, sub_tasks in sub_categories.items():
        sub_scores = []
        existed_sub_tasks = []
        for sub_task in sub_tasks:
            if sub_task in scores:
                existed_sub_tasks.append(sub_task)
                sub_scores.append(float(scores[sub_task]))
        
        if len(existed_sub_tasks) == 0:
            continue
        
        avg_score = sum(sub_scores) / len(sub_scores)
        print(f"{category} - {existed_sub_tasks} - average score: {avg_score:.2f}")

    print(f"\n")
    out_path = f"{path}result.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
