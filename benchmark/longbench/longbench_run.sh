# !/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <budget_ratio> $3 <estimate_ratio> $4 <dtype> $5 <category>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
BUDGET_RATIO=${3}
ESTIMATE_RATIO=${4}
DTYPE=${5}
CATEGORY=${6}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"

if [ "$CATEGORY" == "SQA" ]; then
  tasks=(qasper multifieldqa_en narrativeqa)
elif [ "$CATEGORY" == "MQA" ]; then
  tasks=(hotpotqa 2wikimqa musique dureader)
elif [ "$CATEGORY" == "SUM" ]; then
  tasks=(gov_report qmsum multi_news vcsum)
elif [ "$CATEGORY" == "FSL" ]; then
  tasks=(trec lsht samsum triviaqa)
elif [ "$CATEGORY" == "ST" ]; then
  tasks=(passage_retrieval_en passage_count passage_retrieval_zh)
elif [ "$CATEGORY" == "CC" ]; then
  tasks=(repobench-p lcc)
else
  echo "Unknown CATEGORY: $CATEGORY"
  tasks=()
fi

for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO}"
    bash pred.sh ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO}
done

echo "Start to evaluate..."
python -u eval.py \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \

echo "Results:"
cat "${RESULT_DIR}/result.json"
