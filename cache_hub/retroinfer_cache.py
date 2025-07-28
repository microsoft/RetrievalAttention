import math
import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means
from weighted_flash_decoding import weighted_flash_decoding


# update segment size
THRESHOLD_LENGTH = 1024


class retroinfer_cache(KV_Cache):
    """
    A class representing the KV Cache of RetroInfer.
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
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        n_centroids: int,
        n_segment: int,
        nprobe: int,
        max_compute_cluster_num: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
        self.valid_start = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = cache_unit_size

        self.core = core
        self.dtype = dtype

        self.input_length = self.max_length - max_new_length
        self.max_new_length = min(max_new_length-1, THRESHOLD_LENGTH)   # already generated one token when prefilling
        # used for index update, when exceed THRESHOLD_LENGTH, we need to update the index
        self.input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
        self.n_centroids_per_update_segment = THRESHOLD_LENGTH // 16    # default avg 16 vectors per cluster
        self.n_centroids_per_update_segment = (self.n_centroids_per_update_segment // 32) * 32 # must be divisible by 32
        self.n_centroids_new = ((max_new_length-2) // THRESHOLD_LENGTH) * self.n_centroids_per_update_segment  
        if self.input_length_new > 0:
            self.offload_update_keys = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        # store steady zone
        self.steady_zone_keys = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.static_stride = self.static_pattern_total + self.max_new_length

        # index parameters
        self.n_centroids = n_centroids
        self.n_segment = n_segment
        self.nprobe = nprobe    # retrieve zone size
        self.max_compute_cluster_num = max_compute_cluster_num
        self.es_cluster_num = max_compute_cluster_num - nprobe  # estimation zone size

        # initialize thread pool
        self.thread_pool = ThreadPool(core)
        thread_pool_pointer = self.thread_pool.get()

        # calculate the gpu cache size, buffer size and max total pages for each group
        avg_cluster_size = (self.input_length - self.static_pattern_total) // self.n_centroids
        pages_per_cluster = math.ceil(avg_cluster_size / self.page_size)
        self.cache_size = cache_cluster_num * pages_per_cluster
        # enlarge these values may solve warning and error when decoding
        self.buffer_size = max(int(self.nprobe * 4), 16) * pages_per_cluster

        # whether to pre-allocate GPU buffer and cache before prefilling
        self.allocated = self.pre_allocate_decision()

        # initialize the CPU Wave Buffer
        self.wave_buffer = [WaveBufferCPU(
            self.batch_size, self.kv_head, self.head_dim, self.nprobe, self.page_size, self.n_centroids, 
            # self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_size, self.core, thread_pool_pointer, layer_id)
            # for layer_id in range(self.layer_num)
            self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_size, self.core, thread_pool_pointer)
            for _ in range(self.layer_num)
        ]

        # pin memory indices for hit clusters
        self.hit_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for missing clusters
        self.miss_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for cache update clusters
        self.update_buffer_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]

        # store searched topk cluster ids
        self.cluster_ids = torch.empty((self.batch_size*self.kv_head, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx], self.hit_unit_sizes[ldx], self.hit_unit_sizes_cumsum[ldx], self.hit_num_units[ldx],
                self.miss_unit_idices[ldx], self.miss_unit_sizes[ldx], self.miss_unit_sizes_cumsum[ldx], self.miss_num_units[ldx],
                self.update_buffer_indices[ldx], self.update_unit_sizes[ldx], self.update_cache_indices[ldx], self.update_num_units[ldx], 
                self.cluster_ids
            )

        if self.allocated:
            self.cache_keys = []
            self.cache_values = []
            self.centroids = []
            self.value_sum = []
            self.centroids_mask = []
            self.cluster_size = []
            # allocate GPU Cache data and meta index
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.value_sum.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids_mask.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                                dtype=torch.bool, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cluster_size.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
        else:
            # allocate meta index in CPU
            self.centroids = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.value_sum = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.centroids_mask = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                            dtype=torch.bool, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.cluster_size = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]

        # layer-share cpu pin buffer, transfer gpu keys & values to cpu for segmented k-means
        self.offload_keys = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()
        self.offload_values = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()

        # allocate pin memory to store organized keys & values in CPU
        self.list_keys = []
        self.list_values = []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim), 
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
            self.list_values.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
        self.list_stride = self.input_length-self.static_pattern_total+self.input_length_new
        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_keys, self.offload_values)

        # create multi-streams and events
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()
    
    # decide whether to pre-allocate GPU memory before prefilling
    def pre_allocate_decision(self):
        # estimate the KV Cache GPU memory consumption
        self.esitimate_gpu_memory = 2 * self.layer_num * self.batch_size * self.kv_head * (self.cache_size*self.page_size + self.n_centroids + self.static_pattern_total + self.max_new_length) * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.buffer_size * self.page_size * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        self.esitimate_gpu_memory += 4 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        self.esitimate_gpu_memory /= 1024 * 1024 * 1024
        # print(f"Estimate KV Cache GPU memory consumption: {self.esitimate_gpu_memory:.4f} GB")

        return self.free_memory > self.esitimate_gpu_memory*1.5
    
    # allocate layer-share buffer for computation
    def allocate_computation_buffer(self):
        # execution buffer to store keys & values used to compute attention, shared across layers
        self.execution_buffer_keys = torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                 dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.execution_buffer_values = torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                   dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.valid_lengths = torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, 
                                          device=self.layer_mapping[str(0)]).contiguous()
        self.execution_stride = self.buffer_size * self.page_size + self.static_stride
        
        # allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        
        # allocate layer-share buffer for estimation zone
        self.es_centroids = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                        dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_value_sum = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                         dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_cluster_size = torch.zeros((self.batch_size*self.kv_head, 1, 1, self.es_cluster_num),
                                           dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()

    def prepare_cache(self):
        # sync the last batch of the last layer
        torch.cuda.synchronize()
        self.wave_buffer[self.layer_num-1].construction_sync()
        # clear temp memory
        self.clusters_cpu = None
        self.cluster_size_cpu = None
        self.temp_keys = None
        self.temp_values = None

        if not self.allocated:  # allocate GPU memory after prefilling
            self.cache_keys = []
            self.cache_values = []
            for ldx in range(self.layer_num):
                # allocate GPU Cache data
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                # move meta index to gpu
                self.centroids[ldx] = self.centroids[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.value_sum[ldx] = self.value_sum[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.centroids_mask[ldx] = self.centroids_mask[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.cluster_size[ldx] = self.cluster_size[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
    

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx): 
        """
        Prefill update the key & value cache for per batch for per layer
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            batch_idx: batch index
        """    
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == 1, f"Multi-batch prefilling only support prefill single batch one by one."
        assert seq_len <= self.input_length, f"seq_len({seq_len}) should less than input_length({self.input_length})"
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start[batch_idx]
        valid_length = seq_len - self.static_pattern_total - valid_start

        # sync for the previous layer and batch finish organize pages
        if layer_idx > 0:
            self.wave_buffer[layer_idx-1].construction_sync()
        elif batch_idx > 0: # layer_idx == 0
            self.wave_buffer[self.layer_num-1].construction_sync()
        
        # store in self to avoid deleting when async offload to cpu, shape: (group_num, seq_len, dim)
        self.temp_keys = key_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.temp_values = value_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.mainevents[self.layer_mapping[str(layer_idx)]].record()

        # async offload keys & values to cpu
        with torch.cuda.stream(self.copystream):
            self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
            self.offload_keys[:, :valid_length, :].copy_(self.temp_keys, non_blocking=True)
            self.offload_values[:, :valid_length, :].copy_(self.temp_values, non_blocking=True)
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        # copy steady zone to pre-allocated memory
        self.steady_zone_keys[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            key_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_keys[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            key_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            value_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            value_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)

        # compute key mean, shape (group_num, 1, head_dim)
        mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

        # segmented k-means
        _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
            key=self.temp_keys-mean_key,    # centering to 0
            value=self.temp_values,
            num_centroids=self.n_centroids,
            num_segments=self.n_segment,
        )
        # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids

        # copy meta index
        self.centroids[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_centroids + mean_key)         # (group_num, n_centroids, dim)
        self.value_sum[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_value_sum)                    # (group_num, n_centroids, dim)
        self.centroids_mask[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size == 0)          # (group_num, n_centroids)
        self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size.to(self.dtype))  # (group_num, n_centroids)

        # these data will be used to organize the cpu kv
        self.cluster_size_cpu = _cluster_size.cpu().contiguous()    # (group_num, n_centroids)
        self.clusters_cpu = _clusters.cpu().contiguous()            # (group_num, n_centroids, max_cluster_size)
        
        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += seq_len
        
        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]   # ignore mask tokens, shape: (bsz, seq_len, group_num, dim)

    def sync(
        self,
        layer_idx,
        batch_idx
    ):  
        """
        wait async offloading on copystream -> organize kv
        """
        # wait for offload finish
        self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
        # async organize kv
        self.wave_buffer[layer_idx].async_construction(
            self.clusters_cpu,      # (group_num, n_centroids, max_cluster_size)
            self.cluster_size_cpu,  # (group_num, n_centroids)
            batch_idx
        )


    # update KV cache when generate tokens exceed THRESHOLD_LENGTH
    def _update_kv_cache(self):
        for ldx in range(self.layer_num):
            update_keys = self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            update_values = self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(ldx)]].record()

            # move local window
            self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_keys[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]
            self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_values[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]

            # async offload
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(ldx)]].wait()
                self.offload_update_keys.copy_(update_keys, non_blocking=True)
                self.offload_update_values.copy_(update_values, non_blocking=True)
                self.copyevents[self.layer_mapping[str(ldx)]].record()
            
            # compute key mean, shape (batch_size*group_num, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)
            
            # segmented k-means
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys-mean_key,   # centering to 0, (batch_size*group_num, THRESHOLD_LENGTH, dim)
                value=update_values,        # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                num_centroids=self.n_centroids_per_update_segment,
                num_segments=1,
            )
            _centroids += mean_key
            assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids_per_update_segment

            # append to meta index
            self.centroids[ldx] = torch.cat((self.centroids[ldx], _centroids), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.value_sum[ldx] = torch.cat((self.value_sum[ldx], _value_sum), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.centroids_mask[ldx] = torch.cat((self.centroids_mask[ldx], _cluster_size == 0), dim=1) # (batch_szie*group_num, new_n_centroids)
            self.cluster_size[ldx] = torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1) # (batch_szie*group_num, new_n_centroids)
            assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.n_centroids_per_update_segment

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,           # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                self.offload_update_values,         # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                _clusters.cpu().contiguous(),       # (batch_size*group_num, n_centroids_per_update_segment, max_cluster_size)
                _cluster_size.cpu().contiguous()    # (batch_size*group_num, n_centroids_per_update_segment)
            )
        
        # update n_centroids
        self.n_centroids += self.n_centroids_per_update_segment
        # re-allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        # reset static pattern
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end


    def decode_update_kv_cache(self,
        key_states,         # (bs, length(=1), group_num, dim)
        value_states,       # (bs, length(=1), group_num, dim)
        layer_idx
    ):
        # index update
        if self.static_pattern_total == self.static_pattern_start + self.static_pattern_end + THRESHOLD_LENGTH:
            # print("Updating KV cache ...")
            self._update_kv_cache()
            # print("KV cache updated, continue decoding ...")

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None   # no use the return value
    

    def compute(self, queries, layer_idx):
        """
        queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
        """
        # assert queries.size(0) == self.batch_size
        # assert queries.size(1) == 1
        # assert queries.size(2) == self.kv_head * self.group_size == self.num_heads
        # assert queries.size(3) == self.head_dim

        static_len = self.static_pattern_total if layer_idx == self.layer_num - 1 else self.static_pattern_total + 1

        # search for TopK centroids
        batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                           self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                           self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
        dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
        dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
        cI = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
        self.cluster_ids.copy_(cI[..., :self.nprobe])

        # estimation zone computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(self.centroids[layer_idx], self.es_centroids, 
                                self.value_sum[layer_idx], self.es_value_sum, 
                                self.cluster_size[layer_idx], self.es_cluster_size,
                                cI, self.batch_groups, self.n_centroids, self.es_cluster_num, 
                                self.max_compute_cluster_num, self.nprobe, self.es_cluster_num)
            
            es_out, es_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                self.es_centroids,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_value_sum,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_cluster_size,    # [batch_size*group_num, 1, 1, es_cluster]
                previous_out=None, previous_lse=None,
                return_softmax_lse=True)
        else:
            es_out, es_lse = None, None
        
        # cache access and submit cache update tasks to thread pool
        self.wave_buffer[layer_idx].batch_access()

        # assemble the execution buffer
        gather_copy_and_concat(self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys, 
                               self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values,
                               self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                               self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                               self.valid_lengths, self.batch_groups, 
                               self.static_stride, self.list_stride, self.cache_stride,
                               self.execution_stride, self.buffer_size, static_len)

        # flash attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths,
            return_softmax_lse=False
        )

        # admiss pages from execution buffer to GPU cache
        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        gather_copy_and_scatter(self.execution_buffer_keys, self.cache_keys[layer_idx], self.execution_buffer_values, self.cache_values[layer_idx],
                                self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], self.update_cache_indices[layer_idx], 
                                self.update_num_units[layer_idx], self.batch_groups, self.execution_stride, self.cache_stride,
                                self.buffer_size, static_len)

        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
