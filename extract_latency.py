import re
import os

datas = [[] for _ in range(2)]

for i in range(10):
    for file in os.listdir(f"traces_rep_old_{i}"):
        with open(os.path.join(f"traces_rep_old_{i}", file), 'r') as f:
            content = f.read()
        match = re.search(r"Decoding latency:\s*([\d.]+)\s*ms/step", content)
        if match:
            decoding_latency = float(match.group(1))
            if file.split("_")[3].startswith("1"):
                datas[0].append(decoding_latency)
            else:
                datas[1].append(decoding_latency)
        else:
            print("Decoding latency not found.")
assert len(datas[0]) == len(datas[1])
print(sum(datas[0]) / len(datas[0]), sum(datas[1]) / len(datas[1]))