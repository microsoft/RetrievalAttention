export CUDA_VISIBLE_DEVICES=0
mkdir -p different_tasks_logs


################################ Full Attention ################################
# fwe
for bsz in 1 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --task_name fwe \
            --batch_size $bsz > different_tasks_logs/full_attn_fwe_bsz${bsz}_${round}.log 2>&1
    done
done

# vt
for bsz in 1 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --task_name vt \
            --batch_size $bsz > different_tasks_logs/full_attn_vt_bsz${bsz}_${round}.log 2>&1
    done
done

# qa1
for bsz in 1 4
do
    for round in 1 2
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --task_name qa1 \
            --batch_size $bsz > different_tasks_logs/full_attn_qa1_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# fwe
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --task_name fwe \
            --batch_size $bsz > different_tasks_logs/retroinfer_fwe_bsz${bsz}_${round}.log 2>&1
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
            --task_name fwe \
            --batch_size $bsz > different_tasks_logs/retroinfer_fwe_bsz${bsz}_${round}.log 2>&1
    done
done

# vt
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --task_name vt \
            --batch_size $bsz > different_tasks_logs/retroinfer_vt_bsz${bsz}_${round}.log 2>&1
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
            --task_name vt \
            --batch_size $bsz > different_tasks_logs/retroinfer_vt_bsz${bsz}_${round}.log 2>&1
    done
done

# qa1
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --use_cuda_graph \
            --task_name qa1 \
            --batch_size $bsz > different_tasks_logs/retroinfer_qa1_bsz${bsz}_${round}.log 2>&1
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
            --task_name qa1 \
            --batch_size $bsz > different_tasks_logs/retroinfer_qa1_bsz${bsz}_${round}.log 2>&1
    done
done


unset CUDA_VISIBLE_DEVICES