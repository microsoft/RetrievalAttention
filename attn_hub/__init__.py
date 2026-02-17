
from .full_attn import full_decode_attn, full_prefill_attn
from .retroinfer_attn import retroinfer_decode_attn

try:
    from .minfer import prefill_minfer
except ImportError:
    def prefill_minfer(query_states, key_states, value_states, best_patterns):
        raise ImportError("MInference is not installed, so minfer is not supported.")

try:
    from .xattn import prefill_xattn
except ImportError:
    def prefill_xattn(query_states, key_states, value_states, threshold, causal):
        raise ImportError("Block-Sparse-Attention is not installed, so xattn is not supported.")
