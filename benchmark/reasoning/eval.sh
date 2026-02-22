set -ex

MODEL_NAME_OR_PATH=$1
ATTN_TYPE=$2
DATA_NAME=$3
START=$4
NUM_TEST_SAMPLE=$5
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval


TOKENIZERS_PARALLELISM=false \
# numactl --cpunodebind=0,1 python -u math_eval.py \
python -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split "test" \
    --prompt_type "orz" \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call 32768 \
    --seed 2025 \
    --n_sampling 8 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --start ${START} \
    --end -1 \
    --save_outputs \
    --overwrite \
    --attn_type ${ATTN_TYPE} \
    --do_sample \
    --dtype "bf16" \
    --batch_size 8
