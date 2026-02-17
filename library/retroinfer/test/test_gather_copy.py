import torch
import numpy as np
import time
import math
import random
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, reorganize_vectors, gather_copy_cluster_and_concat_fuse


DTYPE = torch.bfloat16


# generate two random indices with shape (rows, cols)
def gen_two_indices(rows, cols, max_range1, max_range2, unit_size):
    src_indices1 = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size1 = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices1 = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks1 = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    src_indices2 = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size2 = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices2 = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks2 = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    for i in range(rows):
        num = np.random.randint(int(0.8*cols), cols)
        if i == 1:
            num1 = 0
            num2 = num
        elif i == 5:
            num1 = num
            num2 = 0
        else:
            num1 = np.random.randint(0, num)
            num2 = num - num1

        src_indices1[i, :num1] = np.random.choice(max_range1, num1, replace=False)
        cumsum = 0
        for j in range(num1):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            src_copy_size1[i, j] = copy_size
            dst_indices1[i, j] = cumsum
            cumsum += copy_size
        # 设定边界拷贝
        if num1 > 0:
            x = np.random.randint(0, num1)
            src_indices1[i, x] = max_range1+unit_size-src_copy_size1[i, x]
        copy_chunks1[i] = num1

        src_indices2[i, :num2] = np.random.choice(max_range2, num2, replace=False)
        cumsum = 0
        for j in range(num2):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            src_copy_size2[i, j] = copy_size
            dst_indices2[i, j] = cumsum
            cumsum += copy_size
        copy_chunks2[i] = num2

    src_indices1 = torch.from_numpy(src_indices1).pin_memory()
    src_copy_size1 = torch.from_numpy(src_copy_size1).pin_memory()
    dst_indices1 = torch.from_numpy(dst_indices1).pin_memory()
    copy_chunks1 = torch.from_numpy(copy_chunks1).pin_memory()

    src_indices2 = torch.from_numpy(src_indices2).pin_memory()
    src_copy_size2 = torch.from_numpy(src_copy_size2).pin_memory()
    dst_indices2 = torch.from_numpy(dst_indices2).pin_memory()
    copy_chunks2 = torch.from_numpy(copy_chunks2).pin_memory()
    return src_indices1, src_copy_size1, dst_indices1, copy_chunks1, src_indices2, src_copy_size2, dst_indices2, copy_chunks2

def test_concat_gather_copy():
    groups = 8
    src_vector_num1 = 1769
    src_vector_num2 = 12397
    src_unit_num3 = 1000
    buffer_unit_num = 400
    index_length = 400
    unit_size = 8
    dim = 128
    copy_vector_num = 1602
    buffer_vector_num = buffer_unit_num * unit_size + src_vector_num1

    copy_vector_num_tensor = torch.tensor(copy_vector_num, dtype=torch.int32, device='cuda')

    key_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    key_src2 = torch.randn((groups, src_vector_num2, dim), pin_memory=True, dtype=DTYPE).contiguous()
    key_src3 = torch.randn((groups, src_unit_num3, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    value_src2 = torch.randn((groups, src_vector_num2, dim), pin_memory=True, dtype=DTYPE).contiguous()
    value_src3 = torch.randn((groups, src_unit_num3, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    valid_lengths = torch.empty((groups,), dtype=torch.int32, pin_memory=True)

    src_indices1, src_copy_size1, dst_indices1, copy_chunks1, src_indices2, src_copy_size2, dst_indices2, copy_chunks2 = gen_two_indices(groups, index_length, src_vector_num2-unit_size, src_unit_num3, unit_size)
    torch.cuda.synchronize()

    t1 = time.time()
    gather_copy_and_concat(key_src1, key_src2, key_src3, key_dst1,
                           value_src1, value_src2, value_src3, value_dst1,
                           src_indices1, src_copy_size1, dst_indices1, copy_chunks1,
                           src_indices2, src_copy_size2, dst_indices2, copy_chunks2,
                           valid_lengths, groups, src_vector_num1, src_vector_num2, src_unit_num3, 
                           buffer_vector_num, index_length, copy_vector_num_tensor)

    torch.cuda.synchronize()
    print("cuda time: ", time.time()-t1)

    print("valid_lengths: ", valid_lengths)

    for i in range(groups):
        print(f"group{i}, {copy_chunks1[i]}, {copy_chunks2[i]}")

        key_dst2[i, :copy_vector_num, :] = key_src1[i, :copy_vector_num, :]
        value_dst2[i, :copy_vector_num, :] = value_src1[i, :copy_vector_num, :]
        copy_num = copy_vector_num

        for j in range(copy_chunks1[i]):
            key_dst2[i, copy_num:copy_num+src_copy_size1[i, j], :] = key_src2[i, src_indices1[i, j]:src_indices1[i, j]+src_copy_size1[i, j], :]
            value_dst2[i, copy_num:copy_num+src_copy_size1[i, j], :] = value_src2[i, src_indices1[i, j]:src_indices1[i, j]+src_copy_size1[i, j], :]
            copy_num += src_copy_size1[i, j]
        
        for j in range(copy_chunks2[i]):
            key_dst2[i, copy_num:copy_num+src_copy_size2[i, j], :] = key_src3[i, src_indices2[i, j], :src_copy_size2[i, j], :]
            value_dst2[i, copy_num:copy_num+src_copy_size2[i, j], :] = value_src3[i, src_indices2[i, j], :src_copy_size2[i, j], :]
            copy_num += src_copy_size2[i, j]
        
        assert copy_num == valid_lengths[i], f"{i, copy_num, valid_lengths[i]}"
    
    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



def gen_indices(rows, cols, max_range1, max_range2, unit_size):
    src_indices = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    for i in range(rows):
        if i == 1:
            copy_chunks[i] = 0
            continue
        
        num = np.random.randint(int(0.2*cols), int(0.8*cols))
        
        src_indices[i, :num] = np.random.choice(max_range1, num, replace=False)
        dst_indices[i, :num] = np.random.choice(max_range2, num, replace=False)
        for j in range(num):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            if src_indices[i, j] + copy_size > max_range1:  # overflow
                copy_size = max_range1 - src_indices[i, j]
            src_copy_size[i, j] = copy_size
        copy_chunks[i] = num

    src_indices = torch.from_numpy(src_indices).pin_memory()
    src_copy_size = torch.from_numpy(src_copy_size).pin_memory()
    dst_indices = torch.from_numpy(dst_indices).pin_memory()
    copy_chunks = torch.from_numpy(copy_chunks).pin_memory()

    return src_indices, src_copy_size, dst_indices, copy_chunks

def test_gather_copy_scatter():
    groups = 8
    src_unit_num = 400
    dst_unit_num = 1000
    index_length = 400
    unit_size = 8
    dim = 128
    copy_start = 97

    copy_start_tensor = torch.tensor(copy_start, dtype=torch.int32, device='cuda')

    key_src = torch.randn((groups, src_unit_num*unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, dst_unit_num, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src = torch.randn((groups, src_unit_num*unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, dst_unit_num, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    src_indices, src_copy_size, dst_indices, copy_chunks = gen_indices(groups, index_length, src_unit_num*unit_size-copy_start, dst_unit_num, unit_size)
    torch.cuda.synchronize()

    t1 = time.time()
    gather_copy_and_scatter(key_src, key_dst1, value_src, value_dst1, 
                            src_indices, src_copy_size, dst_indices, copy_chunks, 
                            groups, src_unit_num*unit_size, dst_unit_num, index_length, copy_start_tensor)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-t1)

    for i in range(groups):
        print(f"group{i}, {copy_chunks[i]}")
        for j in range(copy_chunks[i]):
            key_dst2[i, dst_indices[i, j], :src_copy_size[i, j], :] = key_src[i, copy_start+src_indices[i, j]:copy_start+src_indices[i, j]+src_copy_size[i, j], :]
            value_dst2[i, dst_indices[i, j], :src_copy_size[i, j], :] = value_src[i, copy_start+src_indices[i, j]:copy_start+src_indices[i, j]+src_copy_size[i, j], :]
    
    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



def test_gather_copy_vectors():
    groups = 8
    src_vector_num = 8192
    dim = 128
    nprobe = 150
    index_size = 2048
    copy_vector_num = index_size - nprobe
    buffer_size = copy_vector_num

    key_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, buffer_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, buffer_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    src_metadata = torch.randn(size=(groups, src_vector_num), dtype=DTYPE, device='cuda').contiguous()
    dst_metadata1 = torch.empty((groups, buffer_size), dtype=DTYPE, device='cuda').contiguous()
    dst_metadata2 = dst_metadata1.clone()

    indices = torch.empty((groups, index_size), dtype=torch.int64, device='cuda')
    for i in range(groups):
        indices[i, :] = torch.randperm(src_vector_num)[:index_size].to(torch.int64).to("cuda")
    
    torch.cuda.synchronize()
    start = time.time()
    gather_copy_vectors(key_src, key_dst1, value_src, value_dst1, src_metadata, dst_metadata1, 
                        indices, groups, src_vector_num, buffer_size, index_size, nprobe, copy_vector_num)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-start)
    
    copy_indices = indices[:, nprobe:nprobe+copy_vector_num]
    for i in range(groups):
        key_dst2[i, :copy_vector_num, :] = key_src[i, copy_indices[i, :], :]
        value_dst2[i, :copy_vector_num, :] = value_src[i, copy_indices[i, :], :]
        dst_metadata2[i, :copy_vector_num] = src_metadata[i, copy_indices[i, :]]

    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()
    assert (dst_metadata1 == dst_metadata2).all()



def split_integer_sum(a, x):
    # 从 [1, a] 中随机选择 x-1 个分割点
    cuts = torch.sort(torch.randint(0, a, (x - 1,), dtype=torch.int32)).values
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([a])])
    return cuts[1:] - cuts[:-1]  # 每份 = 相邻切点之差

def test_reorganize_vectors():
    groups = 8
    dst_vector_num = 102790
    n_centroids = 3511
    dim = 128
    copy_cluster_num = 511
    copy_start = torch.randint(0, n_centroids - copy_cluster_num + 1, (1,), dtype=torch.int32)[0].item()
    # copy_start = 0
    # copy_start = n_centroids - copy_cluster_num
    print(copy_start)

    cluster_sizes = torch.empty((groups, n_centroids), dtype=torch.int32, device='cuda')
    for i in range(groups):
        cluster_sizes[i] = split_integer_sum(dst_vector_num, n_centroids).to(dtype=torch.int32, device='cuda')
    
    cluster_cumsum = torch.cumsum(cluster_sizes, dim=1, dtype=torch.int32)
    assert (cluster_cumsum[:, -1] == dst_vector_num).all(), f"{cluster_cumsum[:, -1]} != {dst_vector_num}"
    cluster_start_indices = cluster_cumsum - cluster_sizes
    
    max_cluster_size = 0
    src_vector_num = 0
    for i in range(groups):
        src_vector_num_group = 0
        for j in range(copy_cluster_num):
            max_cluster_size = max(max_cluster_size, cluster_sizes[i, copy_start+j].item())
            src_vector_num_group += cluster_sizes[i, copy_start+j].item()
        src_vector_num = max(src_vector_num, src_vector_num_group)
    print(max_cluster_size, src_vector_num, (cluster_sizes[:, copy_start:copy_start+copy_cluster_num] == 0).any())

    key_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, dst_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()

    value_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, dst_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()
    
    clusters = torch.randint(-100, 2*src_vector_num, (groups, copy_cluster_num, max_cluster_size), dtype=torch.int32, device='cuda')
    for i in range(groups):
        random_indices = torch.randperm(src_vector_num, dtype=torch.int32, device='cuda')
        start_idx = 0
        for j in range(copy_cluster_num):
            cluster_size = cluster_sizes[i, copy_start+j].item()
            clusters[i, j, :cluster_size] = random_indices[start_idx:start_idx+cluster_size]
            start_idx += cluster_size
    
    torch.cuda.synchronize()
    start = time.time()
    reorganize_vectors(key_src, key_dst1, value_src, value_dst1, clusters, cluster_cumsum, 
                       groups, copy_start)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-start)

    torch.cuda.synchronize()
    start = time.time()
    for i in range(groups):
        for j in range(copy_cluster_num):
            copy_cluster_size = cluster_sizes[i, copy_start+j]
            copy_start_index = cluster_start_indices[i, copy_start+j]
            for k in range(copy_cluster_size):
                key_dst2[i, copy_start_index+k, :] = key_src[i, clusters[i, j, k], :]
                value_dst2[i, copy_start_index+k, :] = value_src[i, clusters[i, j, k], :]
    torch.cuda.synchronize()
    print("torch time: ", time.time()-start)

    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



def test_gather_copy_cluster_and_concat_fuse(nprobe=70):
    groups = 8
    src_vector_num1 = 1769
    src_vector_num2 = 32397
    dim = 128

    copy_vector_num = 1001
    copy_vector_num_tensor = torch.tensor(copy_vector_num, dtype=torch.int32, device='cuda')

    n_centroids = 2019
    cluster_sizes = torch.empty((groups, n_centroids), dtype=torch.int32, device='cuda')
    for i in range(groups):
        cluster_sizes[i] = split_integer_sum(src_vector_num2, n_centroids).to(dtype=torch.int32, device='cuda')
    cluster_cumsum = torch.cumsum(cluster_sizes, dim=-1, dtype=torch.int32)

    index_size = 1787
    select_indices = torch.randint(0, n_centroids, (groups, index_size), dtype=torch.int64, device='cuda')
    select_indices[2, 0] = 0
    select_indices[5, 0] = n_centroids - 1
    nprobe_tensor = torch.tensor(nprobe, dtype=torch.int32, device='cuda')
    
    buffer_vector_num = nprobe * 18 + copy_vector_num
    print(buffer_vector_num)
    key_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    key_src2 = torch.randn((groups, src_vector_num2, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    value_src2 = torch.randn((groups, src_vector_num2, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    valid_lengths = torch.empty((groups,), dtype=torch.int32, device='cuda')
    
    torch.cuda.synchronize()
    t1 = time.time()
    gather_copy_cluster_and_concat_fuse(key_src1, key_src2, key_dst1, value_src1, value_src2, value_dst1,
                                        cluster_cumsum, select_indices, valid_lengths,
                                        groups, src_vector_num1, src_vector_num2, buffer_vector_num, 
                                        nprobe, nprobe_tensor, copy_vector_num_tensor)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-t1)

    print("valid_lengths: ", valid_lengths)

    for i in range(groups):
        key_dst2[i, :copy_vector_num, :] = key_src1[i, :copy_vector_num, :]
        value_dst2[i, :copy_vector_num, :] = value_src1[i, :copy_vector_num, :]
        copy_num = copy_vector_num

        for j in range(nprobe):
            cluster_id = select_indices[i, j].item()
            start_idx = 0 if cluster_id == 0 else cluster_cumsum[i, cluster_id-1].item()
            end_idx = cluster_cumsum[i, cluster_id].item()
            copy_size = end_idx - start_idx

            flag = False
            if copy_num + copy_size > buffer_vector_num:
                print("overflow at group", i, "cluster", j)
                flag = True
                copy_size = max(0, buffer_vector_num - copy_num)

            key_dst2[i, copy_num:copy_num+copy_size, :] = key_src2[i, start_idx:start_idx+copy_size, :]
            value_dst2[i, copy_num:copy_num+copy_size, :] = value_src2[i, start_idx:start_idx+copy_size, :]
            copy_num += copy_size

            if flag: break
        
        assert copy_num == valid_lengths[i], f"{i, copy_num, valid_lengths[i]}"
    
    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



if __name__ == "__main__":
    for i in range(5):
        test_concat_gather_copy()
        test_gather_copy_scatter()
        test_gather_copy_vectors()
        test_reorganize_vectors()
        test_gather_copy_cluster_and_concat_fuse()
        # for i in range(0, 150):
        #     test_gather_copy_cluster_and_concat_fuse(i)
        print("pass")