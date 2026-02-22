# !/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <task_name> $2 <attn_type> $3 <dtype> $4 <budget_ratio> $5 <estimate_ratio>"
    exit 1
fi

NUM_EXAMPLES=-1
MODEL=${1}
TASK=${2}
ATTN_TYPE=${3}
DTYPE=${4}
BUDGET_RATIO=${5}
ESTIMATE_RATIO=${6}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${ATTN_TYPE}"

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
# numactl --cpunodebind=0,1 python -u pred.py \
python -u pred.py \
    --task ${TASK} \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \
    --dtype ${DTYPE} \
    --device auto \
    --retrieval_budget ${BUDGET_RATIO} \
    --estimation_budget ${ESTIMATE_RATIO} \
    --num_examples ${NUM_EXAMPLES}
