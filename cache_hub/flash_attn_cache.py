import torch
from .cache import KV_Cache


class flash_attn_cache(KV_Cache):
    """
    A class representing the KV Cache of Full flash-attn.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, prefill_bsz, num_gpus, model_size)
        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))

        self.valid_start_list = valid_start  # start index of valid tokens for each batch

        self.valid_length = None     # valid seq length of each batch
        self.valid_length_dict = {}  # allocate valid_length on each device

        self.batch_indices_dict = {}
        for device_idx in self.device_list:
            self.batch_indices_dict[device_idx] = torch.arange(self.batch_size, dtype=torch.int32, device=device_idx)
        self.batch_indices = self.batch_indices_dict[self.layer_mapping[str(0)]]

        self.allocated = self.pre_allocate_decision()

        if self.allocated:
            self.key_cache = [
                torch.zeros((self.batch_size, self.max_length, self.kv_head, self.head_dim),
                    device=self.layer_mapping[str(ldx)], dtype=self.dtype
                ) for ldx in range(self.layer_num)
            ]
            self.value_cache = [
                torch.zeros((self.batch_size, self.max_length, self.kv_head, self.head_dim),
                    device=self.layer_mapping[str(ldx)], dtype=self.dtype
                ) for ldx in range(self.layer_num)
            ]
        else:
            self.key_cache = [
                torch.empty((self.batch_size, self.max_length, self.kv_head, self.head_dim),
                    device='cpu', pin_memory=True, dtype=self.dtype
                ) for _ in range(self.layer_num)
            ]
            self.value_cache = [
                torch.empty((self.batch_size, self.max_length, self.kv_head, self.head_dim),
                    device='cpu', pin_memory=True, dtype=self.dtype
                ) for _ in range(self.layer_num)
            ]

        self.copystream = torch.cuda.Stream()
        self.copyevents = {}
        self.KVreadyevents = {}
        for device_idx in self.device_list:
            with torch.cuda.device(device_idx):
                self.copyevents[device_idx] = torch.cuda.Event()
                self.KVreadyevents[device_idx] = torch.cuda.Event()
    
    # decide whether to pre-allocate GPU memory before prefilling
    def pre_allocate_decision(self):
        kv_consumption = (2 * self.layer_num * self.batch_size * self.max_length * self.kv_head * self.head_dim * 2) / 1024 / 1024 / 1024
        return self.free_memory > kv_consumption * 1.3
    
    def move_gpu(self):
        if not self.allocated:
            for ldx in range(self.layer_num):
                self.key_cache[ldx] = self.key_cache[ldx].to(self.layer_mapping[str(ldx)])
                self.value_cache[ldx] = self.value_cache[ldx].to(self.layer_mapping[str(ldx)])

    
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        """
        update part of batches keys and values, start from start_bdx
        Args:
            query_states: (bsz, seq_len, num_heads, head_dim)
            key_states: (bsz, seq_len, kv_head, head_dim)
            value_states: (bsz, seq_len, kv_head, head_dim)
            layer_idx: the index of the layer
            start_bdx: the start index of the batch
        """
        bsz, seq_len, _, _ = key_states.shape
        assert bsz <= self.prefill_bsz, f"Prefilling batch size ({bsz}) should <= {self.prefill_bsz}."
        assert seq_len <= self.max_length, f"Prefilling sequence length ({seq_len}) exceeds max length ({self.max_length})."

        if self.valid_length is None:
            temp_valid_length = torch.from_numpy(seq_len - self.valid_start_list).to(torch.int32)
            for device_idx in self.device_list:
                self.valid_length_dict[device_idx] = temp_valid_length.to(device_idx)
            self.valid_length = self.valid_length_dict[self.layer_mapping[str(0)]]
        
        _valid_start = self.valid_start_list[start_bdx]
        _valid_length = seq_len - _valid_start

        self.KVreadyevents[self.layer_mapping[str(layer_idx)]].record()

        with torch.cuda.stream(self.copystream):
            self.KVreadyevents[self.layer_mapping[str(layer_idx)]].wait()    # wait for KV ready
            self.key_cache[layer_idx][start_bdx:start_bdx+bsz, :_valid_length, :, :].copy_(key_states[:, _valid_start:, :, :], non_blocking=True)
            self.value_cache[layer_idx][start_bdx:start_bdx+bsz, :_valid_length, :, :].copy_(value_states[:, _valid_start:, :, :], non_blocking=True)
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len
        
        return key_states[:, _valid_start:, :, :], value_states[:, _valid_start:, :, :]
    
    def sync(self, layer_idx, start_bdx):
        self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()  # wait for copy done

    
    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        update all batch of the key and value cache for decoding
        Args:
            key_states: (bsz, seq_len(=1), kv_head, head_dim)
            value_states: (bsz, seq_len(=1), kv_head, head_dim)
            layer_idx: the index of the layer
        """
        self.key_cache[layer_idx][self.batch_indices, self.valid_length, :, :] = key_states[:, 0, :, :]
        self.value_cache[layer_idx][self.batch_indices, self.valid_length, :, :] = value_states[:, 0, :, :]
        
        if layer_idx == self.layer_num - 1:
            self.context += 1
            for device_idx in self.device_list:
                self.valid_length_dict[device_idx] += 1
            self.valid_length = self.valid_length_dict[self.layer_mapping[str(layer_idx)]]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
