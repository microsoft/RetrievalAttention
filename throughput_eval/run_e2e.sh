export CUDA_VISIBLE_DEVICES=0
mkdir -p e2e_logs


################################ 512+32K ################################
# Full Attention
for bsz in 1 4 8 10 15
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --attn_type Full_Flash_Attn \
            --task_name AIME \
            --gen_len 32768 \
            --prefill_bsz 8 \
            --batch_size $bsz > e2e_logs/long_output_full_attn_bsz${bsz}_${round}.log 2>&1
    done
done

# vLLM
for bsz in 1 4 8 16
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test_vllm.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --task_name AIME \
            --gen_len 32768 \
            --batch_size $bsz > e2e_logs/long_output_vllm_bsz${bsz}_${round}.log 2>&1
    done
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for bsz in 32 64 100
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test_vllm.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --task_name AIME \
            --gen_len 32768 \
            --batch_size $bsz > e2e_logs/long_output_vllm_bsz${bsz}_${round}.log 2>&1
    done
done
unset PYTORCH_CUDA_ALLOC_CONF

# RetroInfer
for bsz in 1 4 8 16 32 64
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --attn_type RetroInfer \
            --task_name AIME \
            --gen_len 32768 \
            --prefill_bsz 16 \
            --batch_size $bsz > e2e_logs/long_output_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for bsz in 100
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --attn_type RetroInfer \
            --task_name AIME \
            --gen_len 32768 \
            --prefill_bsz 16 \
            --batch_size $bsz > e2e_logs/long_output_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done
unset PYTORCH_CUDA_ALLOC_CONF

# RetroInfer-GPU
for bsz in 1 2 4 8 14
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --attn_type RetroInfer \
            --gpu_only \
            --task_name AIME \
            --gen_len 32768 \
            --prefill_bsz 8 \
            --batch_size $bsz > e2e_logs/long_output_retroinfer_gpu_bsz${bsz}_${round}.log 2>&1
    done
done



################################ 120K+4K ################################
# Full Attention
for bsz in 1 2 3 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 1 \
            --batch_size $bsz > e2e_logs/long_input_full_attn_bsz${bsz}_${round}.log 2>&1
    done
done

# Full Attention + XAttention
for bsz in 1 2 3 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 1 \
            --prefill_method xattn \
            --batch_size $bsz > e2e_logs/xattn_long_input_full_attn_bsz${bsz}_${round}.log 2>&1
    done
done

# vLLM
for bsz in 1 2 4 8 12 16 24 30
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test_vllm.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --batch_size $bsz > e2e_logs/long_input_vllm_bsz${bsz}_${round}.log 2>&1
    done
done

# RetroInfer
for bsz in 1 2 4 8 12 16
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 1 \
            --batch_size $bsz > e2e_logs/long_input_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 24 30
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 1 \
            --batch_size $bsz > e2e_logs/long_input_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done

# RetroInfer + XAttention
for bsz in 1 2 4 8 12 16
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 4 \
            --prefill_method xattn \
            --batch_size $bsz > e2e_logs/xattn_long_input_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 24 30
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 4 \
            --prefill_method xattn \
            --batch_size $bsz > e2e_logs/xattn_long_input_retroinfer_bsz${bsz}_${round}.log 2>&1
    done
done

# RetroInfer-GPU
for bsz in 1 2 3
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --gpu_only \
            --context_len 120000 \
            --task_name NIAH \
            --gen_len 4096 \
            --prefill_bsz 1 \
            --batch_size $bsz > e2e_logs/long_input_retroinfer_gpu_bsz${bsz}_${round}.log 2>&1
    done
done


unset CUDA_VISIBLE_DEVICES