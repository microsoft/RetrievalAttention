import torch


class KV_Cache:
    """
    A class representing the KV Cache.
    """

    def __init__(
        self,
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
        """ Initializes the KV Cache.
        Args:
            layer_num (int)
            batch_size (int)
            num_key_value_heads (int)
            num_heads (int)
            max_length (int)
            head_dim (int)
            dtype (torch.dtype)
            layer_mapping (dict)
            prefill_bsz (int)
            num_gpus (int)
            model_size (int)
        """

        self.layer_num = layer_num
        self.batch_size = batch_size
        self.max_length = max_length
        self.kv_head = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.layer_mapping = layer_mapping
        self.prefill_bsz = min(prefill_bsz, batch_size)
        self.context = 0

        # estimate free GPU memory when prefilling
        self.num_gpus = num_gpus
        self.model_size = model_size
        # total gpu memory
        total_gpu_memory = self.num_gpus * torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        # model weight consumption
        model_weight_consumption = self.model_size * 2
        # prefill consumption for single GPU
        prefill_consumption = self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # hidden
        prefill_consumption += self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # residual
        prefill_consumption += self.prefill_bsz * (self.num_heads + 2*self.kv_head) * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # qkv
        prefill_consumption += 4 * self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # temp
        # free memory during prefill
        self.free_memory = total_gpu_memory - model_weight_consumption - prefill_consumption*self.num_gpus