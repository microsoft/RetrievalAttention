#!/bin/bash

output_dir="traces_1"

while getopts "d:" opt; do
    case $opt in
        d) output_dir=$OPTARG ;;
        *) echo "Usage: $0 [-d output_directory]" >&2
           exit 1 ;;
    esac
done

mkdir -p "$output_dir"

for data_id in 0 1 2 3; do
    for nprobes in 1.5 2; do
        out_file="${output_dir}/data_${data_id}_nprobe_${nprobes}.txt"
        echo "Running: data_id=$data_id, nprobes=$nprobes -> $out_file"
        python -u simple_test.py --batch_size 1 --data_id $data_id --nprobes $nprobes >$out_file
    done
done
