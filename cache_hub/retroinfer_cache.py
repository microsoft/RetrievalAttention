import math
import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means
from weighted_flash_decoding import weighted_flash_decoding


class retroinfer_cache(KV_Cache):
    """
    A class representing the KV Cache of RetroInfer.
    """

    def __init__(
        self,
        valid_start,    # numpy array of valid start positions for each sample in the batch
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
        pages_per_cluster: int,     # 1 cluster = 2 pages = 2 * 8 vectors
        retrieval_budget: float,
        estimation_budget: float,
        cache_ratio: float,         # ratio of cache size to sequence length
        buffer_cluster_num: int,    # number of clusters in the buffer
        use_cuda_graph: bool,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, prefill_bsz, num_gpus, model_size)
        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        self.valid_start_list = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = 8
        avg_cluster_size = pages_per_cluster * self.page_size
        self.UPDATE_SEGMENT = 1024   # update segment size
        self.UPDATE_CENTROIDS = max(round(self.UPDATE_SEGMENT/avg_cluster_size) // 8 * 8, 8)  # must be divisible by 8
        self.UPDATE_NPROBE = max(round(self.UPDATE_CENTROIDS * retrieval_budget), 1)  # update retrieve zone size per segment
        self.UPDATE_ES = max(round(self.UPDATE_CENTROIDS * estimation_budget), 1)  # update estimation zone size per segment

        # whether to build index when prefilling, update index when decoding
        self.input_length = self.max_length - max_new_length
        actual_gen_len = max_new_length - 1  # exclude the first token generated during prefilling
        assert actual_gen_len >= 0, f"Decoding generation length({actual_gen_len}) should be larger than or equal to 0"
        if self.input_length <= 0:
            raise ValueError(f"input length({self.input_length}) should be larger than 0")
        elif self.input_length < self.static_pattern_total + self.UPDATE_SEGMENT:
            # input length is too short, no need to build index during prefilling
            self.build_index_when_prefilling = False
            # update index when decoding, depends on whether input + output length exceed UPDATE_SEGMENT 
            self.will_update_index = (self.input_length-self.static_pattern_total+actual_gen_len) > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.static_pattern_total + self.UPDATE_SEGMENT
                self.list_stride = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_SEGMENT
                self.n_centroids_new = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_CENTROIDS
                self.nprobe_new = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_NPROBE
            else:
                # fall back to full attention, all KV stores in steady zone
                self.static_stride = self.input_length + actual_gen_len
                self.list_stride = 0
                self.n_centroids_new = 0
                self.nprobe_new = 0
        else:
            self.build_index_when_prefilling = True
            # update index when decoding, depends on whether output length exceed UPDATE_SEGMENT
            self.will_update_index = actual_gen_len > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.UPDATE_SEGMENT + self.static_pattern_total
                self.list_stride = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_SEGMENT + self.input_length - self.static_pattern_total
                self.n_centroids_new = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_CENTROIDS
                self.nprobe_new = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_NPROBE
            else: 
                self.static_stride = actual_gen_len + self.static_pattern_total
                self.list_stride = self.input_length - self.static_pattern_total
                self.n_centroids_new = 0
                self.nprobe_new = 0

        # steady zone keys & values
        self.steady_zone_keys = [
            torch.zeros((self.batch_size, self.kv_head, self.static_stride, self.head_dim), 
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)]) 
            for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros((self.batch_size, self.kv_head, self.static_stride, self.head_dim), 
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)]) 
            for ldx in range(self.layer_num)
        ]

        # index parameters
        self.n_segment = n_segment
        self.n_centroids = n_centroids if self.build_index_when_prefilling else 0
        assert self.n_centroids % math.lcm(8, self.n_segment) == 0, \
            f"n_centroids({self.n_centroids}) should be divisible by LCM of 8 and n_segment({self.n_segment})"
        # retrieve zone size (count by clusters)
        self.nprobe = max(round(self.n_centroids*retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids)
        # estimation zone size (count by clusters)
        self.es_cluster_num = min(round(self.n_centroids*estimation_budget), self.n_centroids-self.nprobe)
        # retrieve zone + estimation zone size
        self.max_compute_cluster_num = self.es_cluster_num + self.nprobe
        assert self.max_compute_cluster_num <= self.n_centroids, \
            f"max_compute_cluster_num({self.max_compute_cluster_num}) should <= n_centroids({self.n_centroids})"
        print(f"Initial n_centroids: {self.n_centroids}, nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}")

        # CUDA graphs
        self.use_cuda_graph = use_cuda_graph
        if self.will_update_index:   # need update when decoding
            if self.use_cuda_graph:
                print("Index will be updated during decoding, so CUDA Graph will be disabled.")
            self.use_cuda_graph = False
        elif not self.build_index_when_prefilling:  # not build index during prefilling and not update during decoding
            if self.use_cuda_graph:
                print("Input + output length too small, fall back to full attention, so CUDA Graph will be disabled.")
            self.use_cuda_graph = False
        if self.use_cuda_graph:
            self.topk_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            if self.es_cluster_num > 0:
                self.es_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            self.attn_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            self.update_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]

        # calculate the GPU block cache size and compute buffer size (count by pages)
        cache_cluster_num = round((self.n_centroids + self.n_centroids_new) * cache_ratio) if cache_ratio > 0.0 \
                            else (self.nprobe + self.nprobe_new) * 3
        self.cache_size = cache_cluster_num * pages_per_cluster
        self.buffer_size = max(buffer_cluster_num, (self.nprobe + self.nprobe_new) * 4) * pages_per_cluster
        print(f"Cache pages: {self.cache_size}, Buffer pages: {self.buffer_size}")

        # whether to pre-allocate GPU cache and buffer before prefilling
        self.allocated = self.pre_allocate_decision()

        # initialize thread pool
        self.thread_pool = ThreadPool(core)
        thread_pool_pointer = self.thread_pool.get()
        # initialize the Wave Buffer
        self.wave_buffer = [WaveBufferCPU(
            self.batch_size, self.kv_head, self.head_dim, self.nprobe, self.nprobe_new, self.page_size, 
            self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_size, core, thread_pool_pointer)
            for _ in range(self.layer_num)
        ]

        # pin memory for hit cluster indices (unit == page)
        self.hit_unit_idices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for missing cluster indices (unit == page)
        self.miss_unit_idices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for cache update cluster indices (unit == page)
        self.update_buffer_indices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # store searched TopK cluster IDs
        self.cluster_ids = torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx], self.hit_unit_sizes[ldx], self.hit_unit_sizes_cumsum[ldx], self.hit_num_units[ldx],
                self.miss_unit_idices[ldx], self.miss_unit_sizes[ldx], self.miss_unit_sizes_cumsum[ldx], self.miss_num_units[ldx],
                self.update_buffer_indices[ldx], self.update_unit_sizes[ldx], self.update_cache_indices[ldx], self.update_num_units[ldx], 
                self.cluster_ids
            )

        if self.allocated:  # allocate GPU block cache and meta index
            self.cache_keys, self.cache_values = [], []
            self.centroids, self.value_sum, self.centroids_mask, self.cluster_size = [], [], [], []
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
        else:  # allocate meta index in CPU, will move to GPU after prefilling
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

        # layer-share cpu pin memory, transfer gpu keys & values to cpu for segmented clustering
        if self.build_index_when_prefilling:
            self.offload_keys = torch.empty(
                (self.prefill_bsz*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
                dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_values = torch.empty(
                (self.prefill_bsz*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
                dtype=self.dtype, pin_memory=True
            ).contiguous()

        # layer-share cpu pin memory, offload update keys & values to cpu for segmented clustering
        if self.will_update_index:
            self.offload_update_keys = torch.empty(
                (self.batch_size*self.kv_head, self.UPDATE_SEGMENT, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size*self.kv_head, self.UPDATE_SEGMENT, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()

        # allocate cpu pin memory to store organized keys & values
        self.list_keys, self.list_values = [], []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.list_stride, self.head_dim), 
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
            self.list_values.append(
                torch.empty((self.batch_size, self.kv_head, self.list_stride, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
        
        # set keys & values pointers in the wave buffer
        for ldx in range(self.layer_num):
            if self.build_index_when_prefilling:
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_keys, self.offload_values)
            elif self.will_update_index:
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_update_keys, self.offload_update_values)
            else:
                self.placeholder = torch.empty((self.kv_head, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.placeholder, self.placeholder)

        # create multi-streams and events for async offloading
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        for device_idx in self.device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()
        
        # set decoding attention function
        self.attn_func = self.dense_attention
    

    def pre_allocate_decision(self):
        """Decide whether to pre-allocate GPU cache and buffers before prefilling"""
        # estimate GPU memory consumption for cache and buffers
        self.esitimate_gpu_memory = 2 * self.layer_num * self.batch_size * self.kv_head * (self.cache_size*self.page_size + self.n_centroids + self.static_stride) * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * (self.buffer_size*self.page_size + self.static_stride) * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        self.esitimate_gpu_memory += 6 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        self.esitimate_gpu_memory /= 1024 * 1024 * 1024
        # print(f"Estimate GPU memory consumption for cache and buffers: {self.esitimate_gpu_memory:.4f} GB")
        return self.free_memory > self.esitimate_gpu_memory*1.5
    
    
    def allocate_computation_buffer(self):
        """Allocate layer-share buffers, dict for different GPUs"""
        self.gemm_o_dict, self.softmax_o_dict, self.norm_dict, self.sum_dict, self.dist_dict = {}, {}, {}, {}, {}
        self.cI_dict, self.cV_dict = {}, {}
        self.es_centroids_dict, self.es_value_sum_dict, self.es_cluster_size_dict= {}, {}, {}
        self.execution_buffer_keys_dict, self.execution_buffer_values_dict, self.valid_lengths_dict = {}, {}, {}
        self.static_len_tensor_dict = {}
        if self.use_cuda_graph:
            self.query_buffer_dict = {}
            self.es_out_dict, self.es_lse_dict = {}, {}
            self.attn_out_dict = {}
        
        for device_idx in self.device_list:
            # for batch_gemm_softmax kernel
            self.gemm_o_dict[device_idx] = torch.zeros(
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.softmax_o_dict[device_idx] = torch.zeros(
                (self.batch_size*self.kv_head, self.group_size, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.norm_dict[device_idx] = torch.zeros(
                (self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                device=device_idx, dtype=torch.float32
            ).contiguous()
            self.sum_dict[device_idx] = torch.zeros(
                (self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                device=device_idx, dtype=torch.float32
            ).contiguous()
            self.dist_dict[device_idx] = torch.zeros(
                (self.batch_size*self.kv_head, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            
            # for topk
            self.cI_dict[device_idx] = torch.zeros((self.batch_size*self.kv_head, self.max_compute_cluster_num),
                                                    device=device_idx, dtype=torch.int64).contiguous()
            self.cV_dict[device_idx] = torch.zeros((self.batch_size*self.kv_head, self.max_compute_cluster_num),
                                                    device=device_idx, dtype=self.dtype).contiguous()
            
            # estimation zone
            self.es_centroids_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim), dtype=self.dtype, device=device_idx
            ).contiguous()
            self.es_value_sum_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim), dtype=self.dtype, device=device_idx
            ).contiguous()
            self.es_cluster_size_dict[device_idx] = torch.zeros(
                (self.batch_groups, 1, 1, self.es_cluster_num), dtype=self.dtype, device=device_idx
            ).contiguous()
            
            # execution buffer
            self.execution_buffer_keys_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                dtype=self.dtype, device=device_idx
            ).contiguous()
            self.execution_buffer_values_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                dtype=self.dtype, device=device_idx
            ).contiguous()
            self.valid_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups), dtype=torch.int32, device=device_idx).contiguous()
            self.static_len_tensor_dict[device_idx] = torch.tensor(
                self.static_pattern_total, dtype=torch.int32, device=device_idx)
            
            # allocate buffers used when enable CUDA graphs
            if self.use_cuda_graph:
                self.query_buffer_dict[device_idx] = torch.zeros(
                    (self.batch_groups, 1, self.group_size, self.head_dim), dtype=self.dtype, device=device_idx
                ).contiguous()
                if self.es_cluster_num > 0:
                    self.es_out_dict[device_idx] = torch.zeros(
                        (self.batch_groups, 1, self.group_size, self.head_dim), dtype=self.dtype, device=device_idx
                    ).contiguous()
                    self.es_lse_dict[device_idx] = torch.zeros(
                        (self.batch_groups, self.group_size, 1), dtype=torch.float32, device=device_idx
                    ).contiguous()
                else:
                    self.es_out_dict[device_idx] = None
                    self.es_lse_dict[device_idx] = None
                self.attn_out_dict[device_idx] = torch.zeros(
                    (self.batch_size, 1, self.num_heads, self.head_dim), dtype=self.dtype, device=device_idx
                ).contiguous()
                
        self.execution_stride = self.buffer_size * self.page_size + self.static_stride

        # point to the buffer of current layer's device
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.static_len_tensor = self.static_len_tensor_dict[self.layer_mapping[str(0)]]
        if self.use_cuda_graph:
            self.query_buffer = self.query_buffer_dict[self.layer_mapping[str(0)]]
            self.attn_out = self.attn_out_dict[self.layer_mapping[str(0)]]
        else:
            self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
            self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
            self.norm = self.norm_dict[self.layer_mapping[str(0)]]
            self.sum = self.sum_dict[self.layer_mapping[str(0)]]
            self.dist = self.dist_dict[self.layer_mapping[str(0)]]
            self.cV = self.cV_dict[self.layer_mapping[str(0)]]
            self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
            self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
            self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]
            self.execution_buffer_keys = self.execution_buffer_keys_dict[self.layer_mapping[str(0)]]
            self.execution_buffer_values = self.execution_buffer_values_dict[self.layer_mapping[str(0)]]
            self.valid_lengths = self.valid_lengths_dict[self.layer_mapping[str(0)]]


    def prepare_cache(self):
        """Ensure GPU cache and buffers are allocated before decoding"""
        if self.build_index_when_prefilling:
            # sync the last batch of the last layer
            torch.cuda.synchronize()
            self.wave_buffer[self.layer_num-1].construction_sync()
            # clear temp memory
            self.clusters_cpu, self.cluster_size_cpu = None, None
            self.temp_keys, self.temp_values = None, None
            torch.cuda.empty_cache()

        if not self.allocated:  # allocate GPU cache and buffers after prefilling
            self.cache_keys, self.cache_values = [], []
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                # move meta index to GPU
                self.centroids[ldx] = self.centroids[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.value_sum[ldx] = self.value_sum[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.centroids_mask[ldx] = self.centroids_mask[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.cluster_size[ldx] = self.cluster_size[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
    

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx): 
        """
        Update the key & value cache per layer during prefilling.
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            start_bdx: start batch index
        """    
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz <= self.prefill_bsz, f"Prefilling batch size ({bsz}) should <= {self.prefill_bsz}."
        assert seq_len <= self.input_length, f"seq_len({seq_len}) should <= input_length({self.input_length})"
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start_list[start_bdx]
        
        if self.build_index_when_prefilling:
            # sync for the previous layer and batch finish their page organization
            if layer_idx > 0:
                self.wave_buffer[layer_idx-1].construction_sync()
            elif start_bdx > 0: # layer_idx == 0
                self.wave_buffer[self.layer_num-1].construction_sync()
            
            # store in `self` to avoid deleting when async offload to CPU, shape: (bsz*group_num, seq_len, dim)
            self.temp_keys = key_states[:, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(1, 2).reshape(bsz*self.kv_head, -1, self.head_dim).contiguous()
            self.temp_values = value_states[:, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(1, 2).reshape(bsz*self.kv_head, -1, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(layer_idx)]].record()

            # async offload keys & values to CPU
            valid_length = seq_len - self.static_pattern_total - valid_start
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
                if valid_length == self.offload_keys.shape[1]:
                    self.offload_keys[:bsz*self.kv_head, :, :].copy_(self.temp_keys, non_blocking=True)
                    self.offload_values[:bsz*self.kv_head, :, :].copy_(self.temp_values, non_blocking=True)
                else:   # loop to preserve pinned for fast copy
                    for i in range(bsz*self.kv_head):
                        self.offload_keys[i, :valid_length, :].copy_(self.temp_keys[i], non_blocking=True)
                        self.offload_values[i, :valid_length, :].copy_(self.temp_values[i], non_blocking=True)
                self.copyevents[self.layer_mapping[str(layer_idx)]].record()
            
            # copy steady zone KV
            end_bdx = start_bdx + bsz
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :self.static_pattern_start, :] = \
                key_states[:, valid_start:valid_start+self.static_pattern_start, :, :].transpose(1, 2)
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, self.static_pattern_start:self.static_pattern_total, :] = \
                key_states[:, seq_len-self.static_pattern_end:seq_len, :, :].transpose(1, 2)
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :self.static_pattern_start, :] = \
                value_states[:, valid_start:valid_start+self.static_pattern_start, :, :].transpose(1, 2)
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, self.static_pattern_start:self.static_pattern_total, :] = \
                value_states[:, seq_len-self.static_pattern_end:seq_len, :, :].transpose(1, 2)

            # compute key mean, shape (bsz*group_num, 1, head_dim)
            mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

            # segmented clustering
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=self.temp_keys-mean_key,    # centering to 0
                value=self.temp_values,
                num_centroids=self.n_centroids,
                num_segments=self.n_segment,
            )
            # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids

            # copy meta index
            self.centroids[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :, :].copy_(_centroids + mean_key)         # (bsz*group_num, n_centroids, dim)
            self.value_sum[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :, :].copy_(_value_sum)                    # (bsz*group_num, n_centroids, dim)
            self.centroids_mask[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :].copy_(_cluster_size == 0)          # (bsz*group_num, n_centroids)
            self.cluster_size[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :].copy_(_cluster_size.to(self.dtype))  # (bsz*group_num, n_centroids)

            # cluster results will be used to organize the offload KV cache
            self.cluster_size_cpu = _cluster_size.cpu().contiguous()    # (bsz*group_num, n_centroids)
            self.clusters_cpu = _clusters.cpu().contiguous()            # (bsz*group_num, n_centroids, max_cluster_size)
        else:   # do not build index during prefilling
            assert valid_start == 0, f"Requests in the same batch should have the same length."
            end_bdx = start_bdx + bsz
            # copy input KV to steady zone
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(key_states.transpose(1, 2))
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(value_states.transpose(1, 2))
            
        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len

            if self.build_index_when_prefilling:
                if self.use_cuda_graph:
                    self.attn_func = self.sparse_attention_with_cudagraph
                else:
                    self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]   # ignore mask tokens, shape: (bsz, seq_len, kv_head, dim)

    def sync(self, layer_idx, start_bdx):  
        """Wait async offloading on copystream -> organize KV on wave buffer"""
        if self.build_index_when_prefilling:
            # wait for offload finish
            self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
            # async organize kv
            self.wave_buffer[layer_idx].async_construction(
                self.clusters_cpu,      # (bsz*group_num, n_centroids, max_cluster_size)
                self.cluster_size_cpu,  # (bsz*group_num, n_centroids)
                start_bdx
            )


    def _update_kv_cache(self):
        """Update KV cache when generate tokens exceed UPDATE_SEGMENT"""
        self.nprobe += self.UPDATE_NPROBE
        self.cluster_ids = torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            torch.cuda.set_device(self.layer_mapping[str(ldx)])
            # extract update segment, shape: (batch_size*kv_head, UPDATE_SEGMENT, head_dim)
            update_keys = self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim).contiguous()
            update_values = self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim).contiguous()
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
            
            # compute key mean, shape (batch_size*kv_head, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)
            
            # segmented k-means
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys-mean_key,   # centering to 0, (batch_size*kv_head, UPDATE_SEGMENT, dim)
                value=update_values,        # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                num_centroids=self.UPDATE_CENTROIDS,
                num_segments=1,
            )
            _centroids += mean_key
            # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.UPDATE_CENTROIDS

            # append to meta index
            self.centroids[ldx] = torch.cat((self.centroids[ldx], _centroids), dim=1)  # (batch_size*kv_head, new_n_centroids, dim)
            self.value_sum[ldx] = torch.cat((self.value_sum[ldx], _value_sum), dim=1)  # (batch_size*kv_head, new_n_centroids, dim)
            self.centroids_mask[ldx] = torch.cat((self.centroids_mask[ldx], _cluster_size == 0), dim=1) # (batch_size*kv_head, new_n_centroids)
            self.cluster_size[ldx] = torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1) # (batch_size*kv_head, new_n_centroids)
            # assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.UPDATE_CENTROIDS

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,           # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                self.offload_update_values,         # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                _clusters.cpu().contiguous(),       # (batch_size*kv_head, UPDATE_CENTROIDS, max_cluster_size)
                _cluster_size.cpu().contiguous(),   # (batch_size*kv_head, UPDATE_CENTROIDS)
                self.cluster_ids                    # (batch_size*kv_head, new_nprobe)
            )
        
        # reset current device (layer 0)
        torch.cuda.set_device(self.layer_mapping[str(0)])
        # switch to sparse attention, and update index will disable cudagraph
        assert not self.use_cuda_graph, "CUDA Graph does not support index updating."
        self.attn_func = self.sparse_attention
        
        # update n_centroids, es_cluster_num
        self.n_centroids += self.UPDATE_CENTROIDS
        self.es_cluster_num += self.UPDATE_ES
        self.max_compute_cluster_num += (self.UPDATE_NPROBE + self.UPDATE_ES)
        
        # re-allocate layer-share buffers
        for device_idx in self.device_list:
            self.gemm_o_dict[device_idx] = torch.zeros(
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.softmax_o_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.group_size, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.norm_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.group_size, (self.n_centroids+256-1)//256), 
                device=device_idx, dtype=torch.float32
            ).contiguous()
            self.sum_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.group_size, (self.n_centroids+256-1)//256),
                device=device_idx, dtype=torch.float32
            ).contiguous()
            self.dist_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.n_centroids), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.cI_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.max_compute_cluster_num), device=device_idx, dtype=torch.int64
            ).contiguous()
            self.cV_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.max_compute_cluster_num), device=device_idx, dtype=self.dtype
            ).contiguous()
            self.es_centroids_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim), dtype=self.dtype, device=device_idx
            ).contiguous()
            self.es_value_sum_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim), dtype=self.dtype, device=device_idx
            ).contiguous()
            self.es_cluster_size_dict[device_idx] = torch.zeros(
                (self.batch_groups, 1, 1, self.es_cluster_num), dtype=self.dtype, device=device_idx
            ).contiguous()
        
        # set pointers to current device (layer 0)
        self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
        self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
        self.norm = self.norm_dict[self.layer_mapping[str(0)]]
        self.sum = self.sum_dict[self.layer_mapping[str(0)]]
        self.dist = self.dist_dict[self.layer_mapping[str(0)]]
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.cV = self.cV_dict[self.layer_mapping[str(0)]]
        self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
        self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
        self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]
        
        # reset static pattern length
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        print(f"nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}, max_compute_cluster_num: {self.max_compute_cluster_num}, n_centroids: {self.n_centroids}")


    def decode_update_kv_cache(
        self,
        key_states,         # (bsz, seq_len(=1), group_num, dim)
        value_states,       # (bsz, seq_len(=1), group_num, dim)
        layer_idx
    ):
        # index update when generate tokens exceed UPDATE_SEGMENT
        if self.static_pattern_total == self.static_pattern_start + self.static_pattern_end + self.UPDATE_SEGMENT:
            self._update_kv_cache()

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None   # not use the return value


    def dense_attention(self, queries, layer_idx, static_len):
        """
        Full Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        attn_out = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                previous_out=None, previous_lse=None,
                cache_seqlens=static_len,
                return_softmax_lse=False
            )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)

    
    def sparse_attention(self, queries, layer_idx, static_len):
        """
        Sparse Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)

        # Softmax(QC^T) -> [batch_size*group_num, group_size, n_centroids]
        batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                           self.batch_groups, self.group_size, self.n_centroids, self.head_dim, self.RSQRT_DIM, 0)
        torch.sum(self.softmax_o, dim=1, out=self.dist)  # Merge groups -> [batch_size*group_num, n_centroids]
        self.dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)  # mask empty clusters
        torch.topk(self.dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True, out=(self.cV, self.cI))
        self.cluster_ids.copy_(self.cI[..., :self.nprobe])  # copy the topk cluster ids to the CPU pin memory

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(
                self.centroids[layer_idx], self.es_centroids, 
                self.value_sum[layer_idx], self.es_value_sum, 
                self.cluster_size[layer_idx], self.es_cluster_size,
                self.cI, self.batch_groups, self.n_centroids, self.es_cluster_num, 
                self.max_compute_cluster_num, self.nprobe, self.es_cluster_num
            )
            
            es_out, es_lse = weighted_flash_decoding(
                                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                                self.es_centroids,       # [batch_size*group_num, es_cluster_num, 1, dim]
                                self.es_value_sum,       # [batch_size*group_num, es_cluster_num, 1, dim]
                                self.es_cluster_size,    # [batch_size*group_num, 1, 1, es_cluster_num]
                                previous_out=None, previous_lse=None,
                                return_softmax_lse=True
                            )
        else:
            es_out, es_lse = None, None
        
        # access cache and submit cache update jobs to thread pool
        self.wave_buffer[layer_idx].batch_access()

        # assemble the execution buffer
        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys, 
            self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values,
            self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
            self.valid_lengths, self.batch_groups, self.static_stride, self.list_stride, self.cache_stride, self.execution_stride, 
            self.buffer_size, self.static_len_tensor
        )

        # attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths,  # valid lengths of retrieve zone + steady zone for each group
            return_softmax_lse=False
        )

        # admit pages from execution buffer to GPU block cache
        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        gather_copy_and_scatter(
            self.execution_buffer_keys, self.cache_keys[layer_idx], 
            self.execution_buffer_values, self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], 
            self.update_cache_indices[layer_idx], self.update_num_units[layer_idx], 
            self.batch_groups, self.execution_stride, self.cache_stride,
            self.buffer_size, self.static_len_tensor
        )
        
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)


    def sparse_attention_with_cudagraph(self, queries, layer_idx, static_len):
        """
        Sparse Attention with CUDA graph
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)
        self.query_buffer.copy_(queries.view(self.batch_groups, 1, self.group_size, self.head_dim), non_blocking=True)
        
        # get topk clusters
        self.topk_cudagraphs[layer_idx].replay()
        self.cluster_ids.copy_(self.cI[..., :self.nprobe])  # GPU -> CPU pin memory

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            self.es_cudagraphs[layer_idx].replay()
        
        # access cache and submit cache update jobs to thread pool
        self.wave_buffer[layer_idx].batch_access()

        # compute attention for retrieve zone and steady zone, merge estimation zone results
        self.attn_cudagraphs[layer_idx].replay()

        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        # admit pages from execution buffer to GPU cache
        self.update_cudagraphs[layer_idx].replay()

        return self.attn_out
    

    def capture_cuda_graph(self):
        """Capture CUDA Graph"""
        if not self.use_cuda_graph:
            return
        
        print("Capture CUDA graph ...")
        for layer_idx in range(self.layer_num):
            with torch.cuda.device(self.layer_mapping[str(layer_idx)]):
                capture_stream = torch.cuda.Stream(device=self.layer_mapping[str(layer_idx)])

                # TopK search CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(self.topk_cudagraphs[layer_idx], stream=capture_stream):
                    batch_gemm_softmax(
                        self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                        self.centroids[layer_idx], 
                        self.gemm_o_dict[self.layer_mapping[str(layer_idx)]], 
                        self.norm_dict[self.layer_mapping[str(layer_idx)]], 
                        self.sum_dict[self.layer_mapping[str(layer_idx)]], 
                        self.softmax_o_dict[self.layer_mapping[str(layer_idx)]],
                        self.batch_groups, self.group_size, self.n_centroids, self.head_dim, 
                        self.RSQRT_DIM, 0
                    )
                    torch.sum(self.softmax_o_dict[self.layer_mapping[str(layer_idx)]], dim=1, 
                              out=self.dist_dict[self.layer_mapping[str(layer_idx)]])
                    self.dist_dict[self.layer_mapping[str(layer_idx)]].masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
                    torch.topk(self.dist_dict[self.layer_mapping[str(layer_idx)]], self.max_compute_cluster_num, 
                               dim=-1, largest=True, sorted=True, 
                               out=(self.cV_dict[self.layer_mapping[str(layer_idx)]], self.cI_dict[self.layer_mapping[str(layer_idx)]]))

                # Estimation zone CUDA graph
                if self.es_cluster_num > 0:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self.es_cudagraphs[layer_idx], stream=capture_stream):
                        gather_copy_vectors(
                            self.centroids[layer_idx], self.es_centroids_dict[self.layer_mapping[str(layer_idx)]], 
                            self.value_sum[layer_idx], self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]], 
                            self.cluster_size[layer_idx], self.es_cluster_size_dict[self.layer_mapping[str(layer_idx)]],
                            self.cI_dict[self.layer_mapping[str(layer_idx)]], 
                            self.batch_groups, self.n_centroids, self.es_cluster_num, 
                            self.max_compute_cluster_num, self.nprobe, self.es_cluster_num
                        )
                        # TODO: add output API in this kernel
                        es_out, es_lse = weighted_flash_decoding(
                                            self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                                            self.es_centroids_dict[self.layer_mapping[str(layer_idx)]],
                                            self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]],
                                            self.es_cluster_size_dict[self.layer_mapping[str(layer_idx)]],
                                            previous_out=None, previous_lse=None,
                                            return_softmax_lse=True
                                        )
                        self.es_out_dict[self.layer_mapping[str(layer_idx)]].copy_(es_out, non_blocking=True)
                        self.es_lse_dict[self.layer_mapping[str(layer_idx)]].copy_(es_lse, non_blocking=True)

                # Retrieval and Steady zone CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(self.attn_cudagraphs[layer_idx], stream=capture_stream):
                    gather_copy_and_concat(
                        self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], 
                        self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]], 
                        self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], 
                        self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                        self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                        self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                        self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]], self.batch_groups, self.static_stride, self.list_stride, self.cache_stride,
                        self.execution_stride, self.buffer_size, self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                    )
                    # TODO: add output API in this kernel
                    attn_out = weighted_flash_decoding(
                                    self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                                    self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]],
                                    self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                                    previous_out=self.es_out_dict[self.layer_mapping[str(layer_idx)]],
                                    previous_lse=self.es_lse_dict[self.layer_mapping[str(layer_idx)]],
                                    cache_seqlens=self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]],
                                    return_softmax_lse=False
                                )
                    self.attn_out_dict[self.layer_mapping[str(layer_idx)]].copy_(attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim), non_blocking=True)
                        
                # Cache update CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(self.update_cudagraphs[layer_idx], stream=capture_stream):
                    gather_copy_and_scatter(
                        self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]], self.cache_keys[layer_idx], 
                        self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]], self.cache_values[layer_idx],
                        self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], 
                        self.update_cache_indices[layer_idx], self.update_num_units[layer_idx], 
                        self.batch_groups, self.execution_stride, self.cache_stride, self.buffer_size, 
                        self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                    )
                
                torch.cuda.synchronize()