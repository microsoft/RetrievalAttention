mkdir -p different_models_logs


export CUDA_VISIBLE_DEVICES=0
################################ Full Attention ################################
# Llama-3.1-8B
for bsz in 1 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name meta-llama/Llama-3.1-8B-Instruct \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/full_attn_llama31_bsz${bsz}_${round}.log 2>&1
    done
done

# Qwen-2.5-7B
for bsz in 1 9
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name Qwen/Qwen2.5-7B-Instruct \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/full_attn_qwen_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# Llama-3.1-8B
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name meta-llama/Llama-3.1-8B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_llama31_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name meta-llama/Llama-3.1-8B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_llama31_bsz${bsz}_${round}.log 2>&1
    done
done

# Qwen-2.5-7B
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name Qwen/Qwen2.5-7B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_qwen_bsz${bsz}_${round}.log 2>&1
    done
done

# for bsz in 64
# do
#     for round in 1 2
#     do
#         numactl --cpunodebind=0 --membind=0,1 python -u test.py \
#             --model_name Qwen/Qwen2.5-7B-Instruct \
#             --attn_type RetroInfer \
#             --use_cuda_graph \
#             --context_len 120000 \
#             --task_name NIAH \
#             --batch_size $bsz > different_models_logs/retroinfer_qwen_bsz${bsz}_${round}.log 2>&1
#     done
# done

for bsz in 72
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1,2 python -u test.py \
            --model_name Qwen/Qwen2.5-7B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_qwen_bsz${bsz}_${round}.log 2>&1
    done
done



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
################################ Qwen2.5-72B ################################
# Full Attention
for bsz in 1 8
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name Qwen/Qwen2.5-72B-Instruct \
            --attn_type Full_Flash_Attn \
            --device auto \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/full_attn_qwen72b_bsz${bsz}_${round}.log 2>&1
    done
done


# RetroInfer
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name Qwen/Qwen2.5-72B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --device auto \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_qwen72b_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1,2,3 python -u test.py \
            --model_name Qwen/Qwen2.5-72B-Instruct \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --device auto \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_models_logs/retroinfer_qwen72b_bsz${bsz}_${round}.log 2>&1
    done
done


unset CUDA_VISIBLE_DEVICES