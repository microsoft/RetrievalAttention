import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import TimeoutError

from grader import *
from parser import *
from utils import load_jsonl


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    
    if 'idx' in samples[0]:
        # print(f"length of samples before dedup: {len(samples)}")
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
        # print(f"length of samples after dedup: {len(samples)}")
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
        for param in params:
            try:
                result = math_equal_process(param)
                scores.append(result)
            except TimeoutError as error:
                print(error)
                scores.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(error)
                exit()
            progress_bar.update(1)

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            print(f"Padding sample {i}")
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    print(f"\n-------------\nscores: {score_mat}\n-------------\n")

    # compute overall results
    np_scores = np.array(score_mat)
    avg_scores  = np.round(np_scores.mean() * 100, decimals=1)
    pass_at_1 = np.mean(np.any(np_scores, axis=1))
    pass_at_1 = np.round(pass_at_1 * 100, decimals=1)

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "avg_acc": float(avg_scores),
        "pass@1": float(pass_at_1),
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json
