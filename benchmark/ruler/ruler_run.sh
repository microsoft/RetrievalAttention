#!/bin/bash
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


if [ $# -ne 8 ]; then
    echo "Usage: $0 <model_name> $1 <prefill_method> $2 <attn_type> $3 <context length> $4 <task> $5 <dtype> $6 <budget_ratio> $7 <estimate_ratio>"
    exit 1
fi

# Root Directories
ROOT_DIR="./ruler_eval_result" # the path that stores generated task samples and model predictions.

NUM_SAMPLES=200
MAX_SEQ_LENGTH=${4}
ATTN_TYPE=${3}
DEVICE=auto
BUDGET_RATIO=${7}
ESTIMATE_RATIO=${8}
PREFILL_METHOD=${2}

# Model and Tokenizer
source ruler_config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME})
IFS=":" read MODEL_NAME MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_NAME}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

# Benchmark and Tasks
source ruler_config_tasks.sh
BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${ATTN_TYPE}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
rm -rf ${DATA_DIR}
rm -rf ${PRED_DIR}
mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}

TASK=${5}
python -u data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type ${MODEL_TEMPLATE_TYPE} \
    --num_samples ${NUM_SAMPLES} \
    ${REMOVE_NEWLINE_TAB}

DTYPE=${6}
# numactl --cpunodebind=0,1 python -u pred/call_api.py \
python -u pred/call_api.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --max_len ${MAX_SEQ_LENGTH} \
    --batch_size 1 \
    --data_dir ${DATA_DIR} \
    --save_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --dtype ${DTYPE} \
    --server_type ${MODEL_FRAMEWORK} \
    --device ${DEVICE} \
    --retrieval_budget ${BUDGET_RATIO} \
    --estimation_budget ${ESTIMATE_RATIO} \
    --synthetic_len ${MAX_SEQ_LENGTH} \
    --prefill_method ${PREFILL_METHOD}

python -u eval/evaluate.py \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}
