export CUDA_VISIBLE_DEVICES=0
mkdir -p different_lengths_logs


################################ Full Attention ################################
# 30K
for bsz in 1 2 4 8 16
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 30000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_30k_bsz${bsz}_${round}.log 2>&1
    done
done

# 60K
for bsz in 1 2 4 8
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_60k_bsz${bsz}_${round}.log 2>&1
    done
done

# 120K
for bsz in 1 2 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_120k_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# 30K
for bsz in 1 2 4 8 16 32 64
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 30000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_30k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 128
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 30000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_30k_bsz${bsz}_${round}.log 2>&1
    done
done

# 60K
for bsz in 1 2 4 8 16 32
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_60k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 64
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_60k_bsz${bsz}_${round}.log 2>&1
    done
done

# 120K
for bsz in 1 2 4 8 16
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_120k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_120k_bsz${bsz}_${round}.log 2>&1
    done
done

# 1024K
for bsz in 1 2
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --context_len 1024000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_1024k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 1024000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_1024k_bsz${bsz}_${round}.log 2>&1
    done
done


unset CUDA_VISIBLE_DEVICES