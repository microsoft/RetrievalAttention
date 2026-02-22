from flash_attn import flash_attn_with_kvcache



def full_prefill_attn(query_states, key_states, value_states, causal):

    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        causal=causal
    )
    
    return attn_out



def full_decode_attn(query_states, key_states, value_states, layer_idx, full_attn_cache):

    valid_len = full_attn_cache.valid_length if layer_idx == full_attn_cache.layer_num-1 else full_attn_cache.valid_length+1

    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        cache_seqlens=valid_len
    )
    
    return attn_out
