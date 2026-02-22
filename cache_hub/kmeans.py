import torch
import triton
import triton.language as tl


@triton.jit
def _triton_assign_kernel(
    K, X, S, C, M,  # data, centroids, data_sum, data_cnt, max_idx
    stride_kz, stride_kn, stride_kd,
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    stride_mz, stride_mn,
    num_tokens, num_centroids,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens

    k_ptrs = K + batch_idx * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.) # [BLOCK_N, BLOCK_D]
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for start_k in tl.range(0, num_centroids, BLOCK_K):
        # load centroids
        k_mask = (start_k + offs_k) < num_centroids
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.) # [BLOCK_D, BLOCK_K]   
        # compute inner product              
        ip = tl.dot(k, x).to(tl.float32) # [BLOCK_N, BLOCK_K]
        ip = tl.where(k_mask[None, :], ip, tl.zeros_like(ip) - float("inf"))
        # get max value and index
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True) # [BLOCK_N,]
        tmp_max_idx += start_k
        # update global max value and index
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        # loop to the next block of centroids
        x_ptrs += BLOCK_K * stride_xk

    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(tl.float32), mask=n_mask[:, None], sem='relaxed')
    tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')

@triton.jit
def _triton_update_kernel(
    X, S, C,  # centroids, data_sum, data_cnt
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    num_centroids,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    NORMORLIZE: tl.constexpr,
):
    start_k = tl.program_id(0) * BLOCK_K
    batch_idx = tl.program_id(1)

    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < num_centroids

    x_ptrs = X + batch_idx * stride_xz + offs_k[:, None] * stride_xk + offs_d[None, :] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_k[:, None] * stride_sk + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz + offs_k[:, None] * stride_ck

    s = tl.load(s_ptrs, mask=k_mask[:, None], other=0.) # [BLOCK_K, BLOCK_D]
    c = tl.load(c_ptrs, mask=k_mask[:, None], other=0)
    x_mask = c > 0
    x = s / c
    if NORMORLIZE:
        x /= tl.sqrt(tl.sum(x * x, axis=-1, keep_dims=True))

    tl.store(x_ptrs, x.to(X.type.element_ty), mask=x_mask)

def _triton_k_means_train(
    data: torch.Tensor,             # [batch_size, num_tokens, dim]
    centroids: torch.Tensor,        # [batch_size, num_centroids, dim]
    max_idx: torch.Tensor = None,   # [batch_size, num_tokens]
    normalize_centroids: bool = True,
    return_indices: bool = False,
):
    batch_size, num_tokens, dim = data.shape
    num_centroids = centroids.shape[1]
    data_sum = torch.zeros_like(centroids, dtype=torch.float32)
    data_cnt = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=data.device)
    if max_idx is None:
        max_idx = torch.empty((batch_size, num_tokens), dtype=torch.int32, device=data.device)
    # assert max_idx.shape == (batch_size, num_tokens)
    block_N = 128
    block_K = 64
    _triton_assign_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        data, centroids, data_sum, data_cnt, max_idx,
        data.stride(0), data.stride(1), data.stride(2),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens, num_centroids,
        BLOCK_N=block_N, BLOCK_K=block_K, BLOCK_D=dim,
        num_warps=4, num_stages=2,
    )
    block_K = 128
    _triton_update_kernel[(triton.cdiv(num_centroids, block_K), batch_size, 1)](
        centroids, data_sum, data_cnt,
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        num_centroids,
        BLOCK_K=block_K, BLOCK_D=dim,
        NORMORLIZE=normalize_centroids,
        num_warps=4, num_stages=1,
    )
    if return_indices:
        return centroids, max_idx, data_cnt.max().item()
    return centroids


@triton.jit
def _triton_reverse_index_kernel(
    M, I, C,  # max_idx, clusters, cluster_size
    stride_mz, stride_mn,
    stride_iz, stride_ik, stride_in,
    stride_cz, stride_ck,
    num_tokens,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_tokens

    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn
    i_ptrs = I + batch_idx * stride_iz
    c_ptrs = C + batch_idx * stride_cz

    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)
    cnt = tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')
    tl.store(i_ptrs + max_idx * stride_ik + cnt * stride_in, offs_n, mask=n_mask)

def triton_reverse_index(
    max_idx: torch.Tensor,  # [batch_size, num_tokens]
    num_centroids: int,
    max_cluster_size: int,
):
    batch_size, num_tokens = max_idx.shape
    clusters = torch.zeros((batch_size, num_centroids, max_cluster_size), dtype=torch.int32, device=max_idx.device)
    cluster_size = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=max_idx.device)
    block_N = 128
    _triton_reverse_index_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        max_idx, clusters, cluster_size,
        max_idx.stride(0), 
        max_idx.stride(1),
        clusters.stride(0), clusters.stride(1), clusters.stride(2),
        cluster_size.stride(0), cluster_size.stride(1),
        num_tokens, BLOCK_N=block_N,
        num_warps=4, num_stages=1,
    )
    return clusters, cluster_size


@triton.jit
def _triton_index_add_kernel(
    V, S, M,  # data, sum, max_idx
    stride_vz, stride_vn, stride_vd,
    stride_sz, stride_sk, stride_sd,
    stride_mz, stride_mn,
    num_tokens,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens

    v_ptrs = V + batch_idx * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)
    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)

    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, v.to(S.type.element_ty), mask=n_mask[:, None], sem='relaxed')

def triton_index_add(
    value: torch.Tensor,    # [batch_size, num_tokens, head_dim]
    max_idx: torch.Tensor,  # [batch_size, num_tokens]
    num_centroids: int,
):
    batch_size, num_tokens, dim = value.shape
    value_sum = torch.zeros((batch_size, num_centroids, dim), dtype=torch.float32, device=value.device)
    block_N = 128
    _triton_index_add_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        value, value_sum, max_idx,
        value.stride(0), value.stride(1), value.stride(2),
        value_sum.stride(0), value_sum.stride(1), value_sum.stride(2),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens,
        BLOCK_N=block_N, BLOCK_D=dim,
    )
    return value_sum.to(value.dtype)


def segment_k_means(
    key: torch.Tensor,    # [batch_size*kv_head_num, num_tokens, head_dim]
    value: torch.Tensor,  # [batch_size*kv_head_num, num_tokens, head_dim]
    num_centroids: int,
    num_iters: int = 10,
    num_segments: int = 1
):
    _, num_tokens, head_dim = key.shape

    # initialize centroids uniformly
    centroid_indices = torch.arange(num_centroids, dtype=torch.float32, device=key.device) * (num_tokens / num_centroids)
    centroid_indices += num_tokens / num_centroids / 2
    centroid_indices = centroid_indices.to(torch.int64)
    centroids = torch.index_select(key, dim=1, index=centroid_indices)

    assert num_centroids % num_segments == 0
    num_tokens_per_segment = num_tokens // num_segments
    num_centroids_per_segment = num_centroids // num_segments
    data = key[:, :num_tokens_per_segment * num_segments].reshape((-1, num_tokens_per_segment, head_dim))
    centroids = centroids.reshape((-1, num_centroids_per_segment, head_dim))
    max_idx = torch.empty((data.shape[0], data.shape[1]), dtype=torch.int32, device=data.device)
    for _ in range(num_iters - 1):
        centroids = _triton_k_means_train(data, centroids, max_idx=max_idx, normalize_centroids=True, return_indices=False)

    data = key.reshape((-1, num_tokens, head_dim))
    centroids = centroids.reshape((-1, num_centroids, head_dim))
    centroids, max_idx, max_cluster_size = _triton_k_means_train(data, centroids, normalize_centroids=False, return_indices=True)

    value_sum = triton_index_add(value.reshape((-1, num_tokens, head_dim)), max_idx, num_centroids)
    clusters, cluster_size = triton_reverse_index(max_idx, num_centroids, max_cluster_size)

    # centroids.reshape: (batch_size*kv_head_num, num_centroids, head_dim)
    # value_sum.reshape: (batch_size*kv_head_num, num_centroids, head_dim)
    # clusters.reshape: (batch_size*kv_head_num, num_centroids, max_cluster_size)
    # cluster_size.reshape: (batch_size*kv_head_num, num_centroids)
    return centroids, value_sum, clusters, cluster_size