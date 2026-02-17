# adapted from https://github.com/microsoft/MInference/blob/main/minference/modules/minference_forward.py
import torch
import math
from minference import vertical_slash_sparse_attention, block_sparse_attention, streaming_forward


last_q = 64
arange = torch.arange(last_q, device="cuda:0")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def minference_prefill_kernel(q, k, v, best_pattern):
    head_dim = q.size(-1)
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k) / math.sqrt(head_dim)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    q_len = q.shape[2]
    ty, vertical_size, slash_size, _ = best_pattern

    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)


def prefill_minfer(
    query_states, key_states, value_states,
    best_patterns,
):
    """
    query_states: [batch_size, seq_len, num_heads, head_dim]
    key_states: [batch_size, seq_len, num_kv_heads, head_dim]
    value_states: [batch_size, seq_len, num_kv_heads, head_dim]
    """
    bsz, q_len, head_num, head_dim = query_states.shape
    kv_head_num = key_states.shape[2]
    kv_group_size = head_num // kv_head_num

    output = torch.empty_like(query_states)
    for bdx in range(bsz):
        for head in range(head_num):
            group = head // kv_group_size
            q = query_states[bdx:bdx+1, :, head, :].unsqueeze(1)    # [1, 1, q_len, head_dim]
            k = key_states[bdx:bdx+1, :, group, :].unsqueeze(1)     # [1, 1, q_len, head_dim]
            v = value_states[bdx:bdx+1, :, group, :].unsqueeze(1)   # [1, 1, q_len, head_dim]
            attn_output = minference_prefill_kernel(q, k, v, best_patterns[str(head)])  # [1, 1, q_len, head_dim]
            output[bdx, :, head, :] = attn_output[0, 0, :, :]
    return output
