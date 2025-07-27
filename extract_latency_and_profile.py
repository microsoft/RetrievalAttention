import os
import re
import numpy as np
from collections import defaultdict

prefix = "traces_rep_old_with_time_2"

def extract_metrics_from_file(filepath):
    decoding_latency = None
    insert_times = []
    update_times = []
    evict_times = []

    with open(filepath, 'r') as f:
        for line in f:
            if "Decoding latency" in line:
                match = re.search(r"Decoding latency:\s*([\d.]+)\s*ms", line)
                if match:
                    decoding_latency = float(match.group(1))

            elif "insert:" in line:
                match = re.search(r"insert:\s*(\d+)\s*ns", line)
                if match:
                    insert_times.append(int(match.group(1)))

            elif "update:" in line:
                match = re.search(r"update:\s*(\d+)\s*ns", line)
                if match:
                    update_times.append(int(match.group(1)))

            elif "evict:" in line:
                match = re.search(r"evict:\s*(\d+)\s*ns", line)
                if match:
                    evict_times.append(int(match.group(1)))

    return {
        'file': os.path.basename(filepath),
        'decoding_latency_ms': decoding_latency,
        'insert_avg_ns': np.mean(insert_times) if insert_times else None,
        'update_avg_ns': np.mean(update_times) if update_times else None,
        'evict_avg_ns': np.mean(evict_times) if evict_times else None
    }

# 初始化按 nprobe 分类的结果字典
grouped_results = defaultdict(list)

for i in range(10):
    folder = f"{prefix}_{i}"
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            result = extract_metrics_from_file(filepath)

            # 提取 nprobe 值，例如从 data_3_nprobe_1.5.txt 得到 1.5
            try:
                nprobe_key = filename.split("_")[3].replace(".txt", "")
            except IndexError:
                print(f"Skipping file due to unexpected format: {filename}")
                continue

            grouped_results[nprobe_key].append(result)

# 计算每个 nprobe 分组的平均
for nprobe, group in grouped_results.items():
    decoding_latencies = [r['decoding_latency_ms'] for r in group if r['decoding_latency_ms'] is not None]
    insert_avgs = [r['insert_avg_ns'] for r in group if r['insert_avg_ns'] is not None]
    update_avgs = [r['update_avg_ns'] for r in group if r['update_avg_ns'] is not None]
    evict_avgs = [r['evict_avg_ns'] for r in group if r['evict_avg_ns'] is not None]

    print(f"====== Averages for nprobe = {nprobe} ======")
    print(f"Average Decoding Latency: {np.mean(decoding_latencies):.4f} ms")
    print(f"Average Insert Time:      {np.mean(insert_avgs):.2f} ns")
    print(f"Average Update Time:      {np.mean(update_avgs):.2f} ns")
    print(f"Average Evict Time:       {np.mean(evict_avgs):.2f} ns\n")
