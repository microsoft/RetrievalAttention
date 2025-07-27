#!/bin/bash

output_dir="traces_rep_new_with_time_miss_rate"

while getopts "d:" opt; do
    case $opt in
        d) output_dir=$OPTARG ;;
        *) echo "Usage: $0 [-d output_directory]" >&2
           exit 1 ;;
    esac
done

for id in 0 1 2 3 4 5 6 7 8 9; do
    mkdir -p "${output_dir}_${id}"
    echo running iter ${id}
    for data_id in 0 1 2 3; do
        for nprobes in 1.5 2; do
            out_file="${output_dir}_${id}/data_${data_id}_nprobe_${nprobes}.txt"
            echo "Running: data_id=$data_id, nprobes=$nprobes -> $out_file"
            python -u simple_test.py --batch_size 1 --data_id $data_id --nprobes $nprobes >$out_file
        done
    done
done
