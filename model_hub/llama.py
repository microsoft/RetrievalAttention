import gc
import re
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from .LLM import LLM
from cache_hub import flash_attn_cache, retroinfer_cache, retroinfer_cache_gpu
from attn_hub import full_decode_attn, retroinfer_decode_attn, \
                     full_prefill_attn, prefill_xattn, prefill_minfer
from .xattn_thresholds import llama_31_8b_8_thresholds, llama_3_8b_8_thresholds
from .minfer_patterns import llama_31_8b_best_patterns, llama_3_8b_best_patterns


class LlamaLayer:
    """
    A class representing the Llama layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_llama_layer):
        self.wq = hf_llama_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_llama_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_llama_layer.self_attn.v_proj.weight.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_llama_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        self.gate_proj = hf_llama_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_llama_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_llama_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_llama_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_llama_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_llama_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_llama_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.gate_proj, self.up_proj


class LlamaModel(LLM):
    """
    A class representing the Llama model.
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        tokenizer: AutoTokenizer = None
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        self.config = LlamaConfig.from_pretrained(model_name)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]

        self.init_model()


    def _set_cos_sin_cache(self):
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()*self.attention_scaling, freqs.sin()*self.attention_scaling


    def init_model(self):
        hf_llama = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)

        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'
        
        if self.device_map != "auto":   # single GPU
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})

            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(self.device_map, non_blocking=True)

            self.norm_weight = hf_llama.model.norm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for idx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(idx, device=self.device_map)
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                hf_llama.model.layers[idx] = None

        else:   # multi GPUs
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): f'cuda:{ldx // self.layer_interval}'})

            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)

            self.norm_weight = hf_llama.model.norm.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for ldx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(ldx, device=self.layer_mapping[str(ldx)])
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                hf_llama.model.layers[ldx] = None

        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()

        if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            self.thresholds = [torch.tensor(llama_31_8b_8_thresholds[layer_idx]).to(self.layer_mapping[str(layer_idx)]) 
                               for layer_idx in range(self.num_layers)]
            self.best_patterns = llama_31_8b_best_patterns
        elif self.model_name == "gradientai/Llama-3-8B-Instruct-Gradient-1048k":
            self.thresholds = [torch.tensor(llama_3_8b_8_thresholds[layer_idx]).to(self.layer_mapping[str(layer_idx)]) 
                               for layer_idx in range(self.num_layers)]
            self.best_patterns = llama_3_8b_best_patterns
        else:
            self.thresholds = [torch.ones((self.num_heads,), device=self.layer_mapping[str(layer_idx)])*0.9
                               for layer_idx in range(self.num_layers)]
            self.best_patterns = [{str(head_idx): ["vertical_and_slash", 1000, 6096, 1] for head_idx in range(self.num_heads)}
                                  for layer_idx in range(self.num_layers)]


    def init_kv_cache(self, valid_start, attn_config):
        # collect memory from previous kv_cache
        self.kv_cache = None
        gc.collect()

        llama_config = attn_config
        
        # Init kv cache
        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache = flash_attn_cache(
                valid_start = valid_start,
                layer_num = self.num_layers,
                batch_size = self.batch_size,
                max_length = self.max_new_length + self.input_length,
                num_key_value_heads = self.num_key_value_heads,
                num_heads = self.num_heads,
                head_dim = self.head_dim,
                dtype = self.dtype,
                layer_mapping = self.layer_mapping,
                prefill_bsz = self.prefill_bsz,
                num_gpus = self.num_gpus,
                model_size = int(re.search(r'(\d+)[B]', self.model_name).group(1))
            )
        elif self.attention_type == 'RetroInfer':
            retroinfer_config = llama_config.get(self.attention_type)

            if retroinfer_config['gpu_only'] == True:   # GPU-only version
                self.kv_cache = retroinfer_cache_gpu(
                    valid_start = valid_start,
                    layer_num = self.num_layers,
                    batch_size = self.batch_size,
                    max_length = self.max_new_length + self.input_length,
                    num_key_value_heads = self.num_key_value_heads,
                    num_heads = self.num_heads,
                    head_dim = self.head_dim,
                    dtype = self.dtype,
                    layer_mapping = self.layer_mapping,
                    max_new_length = self.max_new_length,
                    static_pattern_start = retroinfer_config["static_pattern_start"],
                    static_pattern_end = retroinfer_config["static_pattern_end"],
                    core = retroinfer_config["core"],
                    n_centroids = retroinfer_config["n_centroids"],
                    n_segment = retroinfer_config["n_segment"],
                    pages_per_cluster = retroinfer_config["pages_per_cluster"],
                    retrieval_budget = retroinfer_config["retrieval_budget"],
                    estimation_budget = retroinfer_config["estimation_budget"],
                    buffer_cluster_num = retroinfer_config["buffer_cluster_num"],
                    prefill_bsz = self.prefill_bsz,
                    num_gpus = self.num_gpus,
                    model_size = int(re.search(r'(\d+)[B]', self.model_name).group(1))
                )
            else:   # Offload version
                self.kv_cache = retroinfer_cache(
                    valid_start = valid_start,
                    layer_num = self.num_layers,
                    batch_size = self.batch_size,
                    max_length = self.max_new_length + self.input_length,
                    num_key_value_heads = self.num_key_value_heads,
                    num_heads = self.num_heads,
                    head_dim = self.head_dim,
                    dtype = self.dtype,
                    layer_mapping = self.layer_mapping,
                    max_new_length = self.max_new_length,
                    static_pattern_start = retroinfer_config["static_pattern_start"],
                    static_pattern_end = retroinfer_config["static_pattern_end"],
                    core = retroinfer_config["core"],
                    n_centroids = retroinfer_config["n_centroids"],
                    n_segment = retroinfer_config["n_segment"],
                    pages_per_cluster = retroinfer_config["pages_per_cluster"],
                    retrieval_budget = retroinfer_config["retrieval_budget"],
                    estimation_budget = retroinfer_config["estimation_budget"],
                    cache_ratio = retroinfer_config["cache_ratio"],
                    buffer_cluster_num = retroinfer_config["buffer_cluster_num"],
                    use_cuda_graph = retroinfer_config["use_cuda_graph"],
                    prefill_bsz = self.prefill_bsz,
                    num_gpus = self.num_gpus,
                    model_size = int(re.search(r'(\d+)[B]', self.model_name).group(1))
                )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    
    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache.move_gpu()
        elif self.attention_type == 'RetroInfer':
            self.kv_cache.prepare_cache()
        torch.cuda.empty_cache()

    
    def word_embedding(self, inputs_id):
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    
    def lm(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv)
        query_states, key_states, value_states = qkv.split([self.hidden_size, self.hidden_size//self.num_key_value_groups, self.hidden_size//self.num_key_value_groups], dim=-1)
        return query_states, key_states, value_states

    
    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    
    def prefill_attention(self, query_states, key_states, value_states, layer_idx):
        if self.prefill_method == "xattn":
            attn_out = prefill_xattn(query_states, key_states, value_states, self.thresholds[layer_idx], causal=True)
        elif self.prefill_method == "minfer":
            attn_out = prefill_minfer(query_states, key_states, value_states, self.best_patterns[layer_idx])
        else:   # default use full attention
            attn_out = full_prefill_attn(query_states, key_states, value_states, causal=True)
        return attn_out
    

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = full_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == 'RetroInfer':
            attn_out = retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    
    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = (hidden_states.shape[:-1] + (dim,))
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, layer.down_proj)
        return hidden_states 

    
    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:
                self.kv_cache.batch_indices = self.kv_cache.batch_indices_dict[next_device]
                self.kv_cache.valid_length = self.kv_cache.valid_length_dict[next_device]
        elif self.attention_type == 'RetroInfer':
            if hidden_states.shape[1] == 1:
                if isinstance(self.kv_cache, retroinfer_cache_gpu):
                    self.kv_cache.gemm_o = self.kv_cache.gemm_o_dict[next_device]
                    self.kv_cache.softmax_o = self.kv_cache.softmax_o_dict[next_device]
                    self.kv_cache.norm = self.kv_cache.norm_dict[next_device]
                    self.kv_cache.sum = self.kv_cache.sum_dict[next_device]
                    self.kv_cache.dist = self.kv_cache.dist_dict[next_device]
                    self.kv_cache.cI = self.kv_cache.cI_dict[next_device]
                    self.kv_cache.cV = self.kv_cache.cV_dict[next_device]
                    self.kv_cache.es_centroids = self.kv_cache.es_centroids_dict[next_device]
                    self.kv_cache.es_value_sum = self.kv_cache.es_value_sum_dict[next_device]
                    self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size_dict[next_device]
                    self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys_dict[next_device]
                    self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values_dict[next_device]
                    self.kv_cache.valid_lengths = self.kv_cache.valid_lengths_dict[next_device]
                    self.kv_cache.static_len_tensor = self.kv_cache.static_len_tensor_dict[next_device]
                    self.kv_cache.nprobe_tensor = self.kv_cache.nprobe_tensor_dict[next_device]
                else:
                    self.kv_cache.cI = self.kv_cache.cI_dict[next_device]
                    self.kv_cache.static_len_tensor = self.kv_cache.static_len_tensor_dict[next_device]
                    if self.kv_cache.use_cuda_graph:
                        self.kv_cache.query_buffer = self.kv_cache.query_buffer_dict[next_device]
                        self.kv_cache.attn_out = self.kv_cache.attn_out_dict[next_device]
                    else:
                        self.kv_cache.gemm_o = self.kv_cache.gemm_o_dict[next_device]
                        self.kv_cache.softmax_o = self.kv_cache.softmax_o_dict[next_device]
                        self.kv_cache.norm = self.kv_cache.norm_dict[next_device]
                        self.kv_cache.sum = self.kv_cache.sum_dict[next_device]
                        self.kv_cache.dist = self.kv_cache.dist_dict[next_device]
                        self.kv_cache.cV = self.kv_cache.cV_dict[next_device]
                        self.kv_cache.es_centroids = self.kv_cache.es_centroids_dict[next_device]
                        self.kv_cache.es_value_sum = self.kv_cache.es_value_sum_dict[next_device]
                        self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size_dict[next_device]
                        self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys_dict[next_device]
                        self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values_dict[next_device]
                        self.kv_cache.valid_lengths = self.kv_cache.valid_lengths_dict[next_device]
        return hidden_states

    
    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(position_ids, query_states, key_states, self.head_dim, self.cos_sin_cache, True)
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states


    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        return query_states, key_states
