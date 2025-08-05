#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <omp.h>
#include "thread_pool.hpp"

#define PRE_ALLOCATED_NUM 4


class ThreadPool {
    private:
        MyThreadPool* threadpool;
    
    public:
        ThreadPool(int num_threads) {
            threadpool = new MyThreadPool();
            threadpool->Start(static_cast<uint32_t>(num_threads));
        }

        ~ThreadPool() {
            threadpool->Stop();
            delete threadpool;
        }

        MyThreadPool* get() {
            return threadpool;
        }
};


struct ClusterDescriptor {
    bool inBlockCache = false;      // whether the cluster is in the block cache
    int* GPUBlockIDs = nullptr;     // GPU block ids of the cluster, can be incontiguous
    int CPUStartIndex = 0;          // start CPU vector index of the cluster
    int BlockNum = 0;               // number of blocks of the cluster
    int LastBlockSize = 0;          // valid vector number of the last block (only last block may be not full)
    std::list<int64_t>::iterator LRUEntryPointer; // pointer for the cluster in the LRU list
};


class BufferManager {
private:
    const int capacity;             // number of total blocks for LRU Cache
    const int nprobe;               // max number for each batch access
    const int block_size;           // full vector number for one block
    const int max_consider_block;   // max consider block number for each group
    ClusterDescriptor* cluster_descriptors; // cluster descriptors
    std::unordered_set<int> free_block_ids; // free block ids
    std::list<int64_t> lru_keys;            // LRU list for recently used keys

    int miss_num = 0;                   // number of missing keys
    int hit_num = 0;                    // number of hit keys
    int64_t* _miss_keys = nullptr;      // missing keys
    int64_t* _hit_keys = nullptr;       // hit keys

    inline void removeLeastRecentlyUsed() noexcept {
        if (lru_keys.empty()) return;

        // find the least recently used key
        const int64_t lru_key = lru_keys.back();
        lru_keys.pop_back();

        // collect its block ids
        int* block_ids = cluster_descriptors[lru_key].GPUBlockIDs;
        free_block_ids.insert(block_ids, block_ids + cluster_descriptors[lru_key].BlockNum);

        // set to miss
        cluster_descriptors[lru_key].inBlockCache = false;
    }

public:
    BufferManager(int capacity, int nprobe, int block_size, int max_consider_block, ClusterDescriptor* cluster_descriptors)
     : capacity(capacity), nprobe(nprobe), block_size(block_size), max_consider_block(max_consider_block),
     cluster_descriptors(cluster_descriptors) {
        // set free block ids
        free_block_ids.reserve(capacity);
        for (int i = 0; i < capacity; ++i) {
            free_block_ids.insert(i);
        }

        _miss_keys = new int64_t[nprobe];
        _hit_keys = new int64_t[nprobe];
        miss_num = 0;
        hit_num = 0;
    }

    ~BufferManager() {
        free_block_ids.clear();
        lru_keys.clear();
        if (_miss_keys != nullptr) delete[] _miss_keys;
        if (_hit_keys != nullptr) delete[] _hit_keys;
        cluster_descriptors = nullptr;
    }

    inline std::tuple<int, int> batch_update(
        int* update_block_ids, int* update_block_sizes, int* update_block_sizes_cumsum
    ) noexcept {
        // reverse iterate over the hit keys and update the LRU order.
        for (int i = hit_num - 1; i >= 0; --i) {
            const int64_t& key = _hit_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];

            // update LRU order
            lru_keys.erase(cluster_descriptor.LRUEntryPointer);
            lru_keys.push_front(key);
            cluster_descriptor.LRUEntryPointer = lru_keys.begin();
        }

        int admiss_num = 0;             // number of admissible keys
        int total_blocks_needed = 0;    // total number of blocks needed for cache update
        // iterate sequentially over the miss keys and filter out keys that exceed the capacity.
        for (int i = 0; i < miss_num; ++i) {
            const int64_t& key = _miss_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];

            if (total_blocks_needed + cluster_descriptor.BlockNum <= capacity) {
                admiss_num++;
                total_blocks_needed += cluster_descriptor.BlockNum;
            } else {
                // exceed capacity, drop the following keys
                break;
            }
        }

        if (admiss_num == 0) {
            // reset
            miss_num = 0;
            hit_num = 0;
            return { 0, 0 };
        }

        // evict to ensure the have enough space for the admissible keys
        while (free_block_ids.size() < static_cast<size_t>(total_blocks_needed)) {
            removeLeastRecentlyUsed();
        }

        // insert the admissible keys into the cache
        int update_block_num = 0;
        int update_cumsum = 0;
        for (int i = 0; i < admiss_num; ++i) {
            const int64_t& key = _miss_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];
            
            cluster_descriptor.inBlockCache = true;

            auto free_it = free_block_ids.begin();
            for (int j = 0; j < cluster_descriptor.BlockNum - 1; ++j) {
                const int block_id = *free_it;
                free_block_ids.erase(free_it++);
                cluster_descriptor.GPUBlockIDs[j] = block_id;
                update_block_ids[update_block_num] = block_id;
                update_block_sizes[update_block_num] = block_size;
                update_block_sizes_cumsum[update_block_num] = update_cumsum;
                update_cumsum += block_size;
                update_block_num++;
            }
            // last block
            const int block_id = *free_it;
            free_block_ids.erase(free_it++);
            cluster_descriptor.GPUBlockIDs[cluster_descriptor.BlockNum - 1] = block_id;
            update_block_ids[update_block_num] = block_id;
            update_block_sizes[update_block_num] = cluster_descriptor.LastBlockSize;
            update_block_sizes_cumsum[update_block_num] = update_cumsum;
            update_cumsum += cluster_descriptor.LastBlockSize;
            update_block_num++;

            // update LRU order
            lru_keys.push_front(key);
            cluster_descriptor.LRUEntryPointer = lru_keys.begin();
        }

        // if (update_block_num != total_blocks_needed) {
        //     throw std::runtime_error("Update block ids size mismatch!");
        // }

        // reset
        miss_num = 0;
        hit_num = 0;

        return { admiss_num, update_block_num };
    }

    inline std::tuple<int, int, int, int> batch_access(
        const int64_t* keys, const int num, 
        int* hit_block_ids, int* hit_block_sizes, int* hit_block_sizes_cumsum,
        int* miss_block_ids, int* miss_block_sizes, int* miss_block_sizes_cumsum
    ) noexcept {
        if (num == 0) {
            return { 0, 0, 0, 0 };
        }

        miss_num = 0;
        hit_num = 0;
        int hit_block_num = 0;
        int miss_block_num = 0;
        int hit_cumsum = 0;
        int miss_cumsum = 0;
        int consider_block_num = 0;

        for (int i = 0; i < num; ++i) {
            const int64_t& key = keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];
            
            consider_block_num += cluster_descriptor.BlockNum;
            // ignore clusters that can not fit in the buffer
            if (consider_block_num > max_consider_block) {
                printf("Warning: retrieved pages exceeds max consider pages, will skip the remaining clusters, please increase buffer_size to solve this.\n");
                break;
            }
            
            // miss keys, copy from CPU
            if (!cluster_descriptor.inBlockCache) {
                _miss_keys[miss_num++] = key;
                int CPUStartID = cluster_descriptor.CPUStartIndex;
                for (int j = 0; j < cluster_descriptor.BlockNum - 1; ++j) {
                    miss_block_ids[miss_block_num] = CPUStartID;
                    miss_block_sizes[miss_block_num] = block_size;
                    miss_block_sizes_cumsum[miss_block_num] = miss_cumsum;
                    miss_cumsum += block_size;
                    CPUStartID += block_size;
                    miss_block_num++;
                }
                // last block
                miss_block_ids[miss_block_num] = CPUStartID;
                miss_block_sizes[miss_block_num] = cluster_descriptor.LastBlockSize;
                miss_block_sizes_cumsum[miss_block_num] = miss_cumsum;
                miss_cumsum += cluster_descriptor.LastBlockSize;
                miss_block_num++;
            // hit keys, copy from GPU Cache
            } else {
                _hit_keys[hit_num++] = key;

                int* GPUBlockIDs = cluster_descriptor.GPUBlockIDs;
                std::copy(GPUBlockIDs, GPUBlockIDs + cluster_descriptor.BlockNum, hit_block_ids + hit_block_num);
                for (int j = 0; j < cluster_descriptor.BlockNum - 1; ++j) {
                    hit_block_sizes[hit_block_num] = block_size;
                    hit_block_sizes_cumsum[hit_block_num] = hit_cumsum;
                    hit_cumsum += block_size;
                    hit_block_num++;
                }
                // last block
                hit_block_sizes[hit_block_num] = cluster_descriptor.LastBlockSize;
                hit_block_sizes_cumsum[hit_block_num] = hit_cumsum;
                hit_cumsum += cluster_descriptor.LastBlockSize;
                hit_block_num++;
            }
        }

        return { hit_num, miss_num, hit_block_num, miss_block_num };
    }
};


class WaveBufferCPU {
private:
    const int batch_size;   // batch size
    const int group_num;    // kv_head_num
    int batch_groups;
    const int dim;          // dimension of the vector

    const int nprobe;       // searched cluster number
    const int block_size;   // full vector number for one block
    int n_centroids;        // current number of clusters for each group
    const int final_n_centroids;  // final number of clusters for each group (since index may insert new data during decoding)

    const int buffer_size;  // max consider block number for each group
    const int capacity;     // cache capacity

    int num_threads;        // thread number
    int group_per_thread;   // groups per thread

    MyThreadPool* pool_;                    // thread pool
    std::vector<BufferManager*> caches;     // Buffer manager (LRU)

    ClusterDescriptor* cluster_descriptors; // cluster descriptors, (batch_size*group_num, final_n_centroids)

    // (group_num, n_centroids), cumulative sum of the vector number of each cluster in each group
    int* ivf_list_size_csum_array;

    // pointer to the retrieved clusters, [group_num, nprobe]
    int64_t* searched_clusters_ptr;         

    // input data to re-organize keys & values based on the clustering results
    uint16_t* input_key_ptr;        // (groups, input_seq_length, dim)
    uint16_t* input_value_ptr;      // (groups, input_seq_length, dim)
    int* clusters_ptr;              // key id of each cluster in single batch, (groups, n_centroids, max_cluster_size)
    int* cluster_size_ptr;          // cluster size of each cluster in single batch, (groups, n_centroids)
    int64_t input_seq_length;
    int64_t max_cluster_size;

    int current_batch = 0;          // current processing batch index

    // last sequence length of each group, [batch_size*group_num]
    int* last_seq_lengths = nullptr;
    // last number of clusters (before update)
    int last_n_centroids;

public:
    // output indices buffer    
    int* hit_block_ids;             // cache block ids of hit keys
    int* hit_block_sizes;           // valid vector number of each hiy blocks
    int* hit_block_sizes_cumsum;    // cumsum of hit_block_sizes
    int* hit_block_nums;            // number of hit block ids for each group
    
    int* miss_block_ids;            // cpu list block ids of missing keys
    int* miss_block_sizes;          // valid vector number of each missing blocks
    int* miss_block_sizes_cumsum;   // cumsum of miss_block_sizes
    int* miss_block_nums;           // number of missing block ids for each group
        
    int* update_buffer_indices;     // buffer start vector position of each update blocks
    int* update_block_sizes;        // valid vector number of each update blocks
    int* update_cache_indices;      // cache update block ids
    int* update_block_nums;         // number of update block ids for each group

    // output kv buffer
    uint16_t* ivf_key_array;        // cluster keys, (batch_size, group_num, output_seq_length, dim)
    uint16_t* ivf_value_array;      // cluster values, (batch_size, group_num, output_seq_length, dim)
    int64_t output_seq_length;


    WaveBufferCPU(int batch_size, int group_num, int dim, int nprobe, int block_size, 
        int n_centroids, int final_n_centroids, int buffer_size, int capacity, int threads, MyThreadPool* pool)
     : batch_size(batch_size), group_num(group_num), dim(dim), nprobe(nprobe), block_size(block_size),
     n_centroids(n_centroids), final_n_centroids(final_n_centroids), buffer_size(buffer_size), capacity(capacity), pool_(pool) {
        batch_groups = batch_size * group_num;
        // count valid threads
        int min_group_per_thread = 2;
        num_threads = std::min(threads, (batch_groups + min_group_per_thread - 1) / min_group_per_thread);
        group_per_thread = (batch_groups + num_threads - 1) / num_threads;

        ivf_key_array = nullptr;
        ivf_value_array = nullptr;
        
        current_batch = 0;
        
        input_key_ptr = nullptr;
        input_value_ptr = nullptr;
        clusters_ptr = nullptr;
        cluster_size_ptr = nullptr;

        searched_clusters_ptr = nullptr;

        hit_block_ids = nullptr;
        hit_block_sizes = nullptr;
        hit_block_sizes_cumsum = nullptr;
        hit_block_nums = nullptr;

        miss_block_ids = nullptr;
        miss_block_sizes = nullptr;
        miss_block_sizes_cumsum = nullptr;
        miss_block_nums = nullptr;

        update_buffer_indices = nullptr;
        update_block_sizes = nullptr;
        update_cache_indices = nullptr;
        update_block_nums = nullptr;

        last_seq_lengths = new int[batch_groups];
        std::fill(last_seq_lengths, last_seq_lengths + batch_groups, 0);
        last_n_centroids = n_centroids;

        ivf_list_size_csum_array = new int[group_num * n_centroids];

        // store descriptor of each cluster
        cluster_descriptors = new ClusterDescriptor[batch_groups * final_n_centroids];
        for (int i = 0; i < batch_groups * final_n_centroids; ++i) {
            cluster_descriptors[i].GPUBlockIDs = new int[PRE_ALLOCATED_NUM];    // pre-allocate memory
        }

        // prepare caches
        caches.resize(batch_groups, nullptr);
        for (int i = 0; i < batch_groups; ++i) {
            caches[i] = new BufferManager(capacity, nprobe, block_size, buffer_size,
                                          cluster_descriptors + i * final_n_centroids);
        }
    }

    ~WaveBufferCPU() {
        for (int i = 0; i < batch_groups; ++i) {
            if (caches[i] != nullptr) {
                delete caches[i];
                caches[i] = nullptr;
            }
        }
        caches.clear();
        
        if (cluster_descriptors != nullptr) {
            for (int i = 0; i < batch_groups * final_n_centroids; ++i) {
                if (cluster_descriptors[i].GPUBlockIDs != nullptr) {
                    delete[] cluster_descriptors[i].GPUBlockIDs;
                    cluster_descriptors[i].GPUBlockIDs = nullptr;
                }
            }
            delete[] cluster_descriptors;
            cluster_descriptors = nullptr;
        }

        if (last_seq_lengths != nullptr) {
            delete[] last_seq_lengths;
            last_seq_lengths = nullptr;
        }

        if (ivf_list_size_csum_array != nullptr) {
            delete[] ivf_list_size_csum_array;
            ivf_list_size_csum_array = nullptr;
        }

        ivf_key_array = nullptr;
        ivf_value_array = nullptr;

        input_key_ptr = nullptr;
        input_value_ptr = nullptr;
        clusters_ptr = nullptr;
        cluster_size_ptr = nullptr;

        searched_clusters_ptr = nullptr;

        hit_block_ids = nullptr;
        hit_block_sizes = nullptr;
        hit_block_sizes_cumsum = nullptr;
        hit_block_nums = nullptr;

        miss_block_ids = nullptr;
        miss_block_sizes = nullptr;
        miss_block_sizes_cumsum = nullptr;
        miss_block_nums = nullptr;

        update_buffer_indices = nullptr;
        update_block_sizes = nullptr;
        update_cache_indices = nullptr;
        update_block_nums = nullptr;
    }


    void set_indices(
        torch::Tensor& hit_block_ids_tensor,
        torch::Tensor& hit_block_sizes_tensor,
        torch::Tensor& hit_block_sizes_cumsum_tensor,
        torch::Tensor& hit_block_nums_tensor,

        torch::Tensor& miss_block_ids_tensor,
        torch::Tensor& miss_block_sizes_tensor,
        torch::Tensor& miss_block_sizes_cumsum_tensor,
        torch::Tensor& miss_block_nums_tensor,

        torch::Tensor& update_buffer_indices_tensor,
        torch::Tensor& update_block_sizes_tensor,
        torch::Tensor& update_cache_indices_tensor,
        torch::Tensor& update_block_nums_tensor,

        torch::Tensor& searched_clusters
    ) {
        hit_block_ids = static_cast<int*>(hit_block_ids_tensor.data_ptr<int32_t>());
        hit_block_sizes = static_cast<int*>(hit_block_sizes_tensor.data_ptr<int32_t>());
        hit_block_sizes_cumsum = static_cast<int*>(hit_block_sizes_cumsum_tensor.data_ptr<int32_t>());
        hit_block_nums = static_cast<int*>(hit_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(hit_block_ids_tensor.size(-1) == buffer_size, "Wrong hit block ids size.");
        // AT_ASSERT(hit_block_sizes_tensor.size(-1) == buffer_size, "Wrong hit block sizes size.");
        // AT_ASSERT(hit_block_sizes_cumsum_tensor.size(-1) == buffer_size, "Wrong hit block sizes cumsum size.");

        miss_block_ids = static_cast<int*>(miss_block_ids_tensor.data_ptr<int32_t>());
        miss_block_sizes = static_cast<int*>(miss_block_sizes_tensor.data_ptr<int32_t>());
        miss_block_sizes_cumsum = static_cast<int*>(miss_block_sizes_cumsum_tensor.data_ptr<int32_t>());
        miss_block_nums = static_cast<int*>(miss_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(miss_block_ids_tensor.size(-1) == buffer_size, "Wrong miss block ids size.");
        // AT_ASSERT(miss_block_sizes_tensor.size(-1) == buffer_size, "Wrong miss block sizes size.");
        // AT_ASSERT(miss_block_sizes_cumsum_tensor.size(-1) == buffer_size, "Wrong miss block sizes cumsum size.");

        update_buffer_indices = static_cast<int*>(update_buffer_indices_tensor.data_ptr<int32_t>());
        update_block_sizes = static_cast<int*>(update_block_sizes_tensor.data_ptr<int32_t>()); 
        update_cache_indices = static_cast<int*>(update_cache_indices_tensor.data_ptr<int32_t>());
        update_block_nums = static_cast<int*>(update_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(update_buffer_indices_tensor.size(-1) == buffer_size, "Wrong update buffer indices size.");
        // AT_ASSERT(update_block_sizes_tensor.size(-1) == buffer_size, "Wrong update block sizes size.");
        // AT_ASSERT(update_cache_indices_tensor.size(-1) == capacity, "Wrong update cache indices size.");

        searched_clusters_ptr = static_cast<int64_t*>(searched_clusters.data_ptr<int64_t>());
        // AT_ASSERT(searched_clusters.size(-1) == nprobe, "Wrong searched clusters size.");
    }

    void set_kv(
        torch::Tensor& ivf_key_tensor,      // (bsz, groups, seq_len, dim)
        torch::Tensor& ivf_value_tensor,    // (bsz, groups, seq_len, dim)
        torch::Tensor& input_keys,          // (groups, seq_len, dim)
        torch::Tensor& input_values         // (groups, seq_len, dim)
    ) {
        if (ivf_key_tensor.dtype() == torch::kFloat16) {     // fp16
            ivf_key_array = reinterpret_cast<uint16_t*>(ivf_key_tensor.data_ptr<at::Half>());
            ivf_value_array = reinterpret_cast<uint16_t*>(ivf_value_tensor.data_ptr<at::Half>());
            input_key_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::Half>());
            input_value_ptr = reinterpret_cast<uint16_t*>(input_values.data_ptr<at::Half>());
        } else {    // bf16
            ivf_key_array = reinterpret_cast<uint16_t*>(ivf_key_tensor.data_ptr<at::BFloat16>());
            ivf_value_array = reinterpret_cast<uint16_t*>(ivf_value_tensor.data_ptr<at::BFloat16>());
            input_key_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::BFloat16>());
            input_value_ptr = reinterpret_cast<uint16_t*>(input_values.data_ptr<at::BFloat16>());
        }
        output_seq_length = ivf_key_tensor.size(2);
        input_seq_length = input_keys.size(1);
    }



    // compute the cumsum of the block number of each cluster in group idx
    // initialize the cluster descriptors
    int parse_clusters(
        int* lists_size,    // shape (n_centroids,), represent the length of each cluster
        const int idx
    ) {
        // cumsum of the cluster size of each cluster
        int* list_size_csum = ivf_list_size_csum_array + idx * static_cast<int64_t>(n_centroids);
        // cluster descriptor of this group
        auto cluster_descriptors_group = cluster_descriptors + (current_batch * group_num + idx) * static_cast<int64_t>(final_n_centroids);

        int total_size = 0;
        for (int i = 0; i < n_centroids; ++i) {
            int list_size = lists_size[i];
            // calculate the number of blocks for this cluster
            const int block_num = (list_size + block_size - 1) / block_size;

            // initialize the cluster descriptor
            cluster_descriptors_group[i].inBlockCache = false;
            if (block_num > PRE_ALLOCATED_NUM) {
                delete[] cluster_descriptors_group[i].GPUBlockIDs;
                cluster_descriptors_group[i].GPUBlockIDs = new int[block_num];  // allocate larger memory
            }
            cluster_descriptors_group[i].CPUStartIndex = total_size;
            cluster_descriptors_group[i].BlockNum = block_num;
            cluster_descriptors_group[i].LastBlockSize = list_size % block_size == 0 ? block_size : list_size % block_size;

            total_size += list_size;
            list_size_csum[i] = total_size;
        }
        return total_size;
    }

    // organize the KV for each group
    void organize_kv(void* para) {
        int idx = reinterpret_cast<std::intptr_t>(para);

        int* ivf_list_size_csum = ivf_list_size_csum_array + idx * static_cast<int64_t>(n_centroids);
        int* invlists = clusters_ptr + idx * n_centroids * max_cluster_size;
        int* invlists_size = cluster_size_ptr + idx * n_centroids;
        uint16_t* values = input_value_ptr + idx * input_seq_length * dim;
        uint16_t* keys = input_key_ptr + idx * input_seq_length * dim;
        uint16_t* parse_key = ivf_key_array + (current_batch * group_num + idx) * output_seq_length * dim;
        uint16_t* parse_value = ivf_value_array + (current_batch * group_num + idx) * output_seq_length * dim;

        // organize keys & values by clusters
        for (int i = 0; i < n_centroids; ++i) {
            int list_size = invlists_size[i];  // actual list size, count by vectors
            const int start_idx = i == 0 ? 0 : ivf_list_size_csum[i-1]; // start index of this list in the parse buffer

            for (int j = 0; j < list_size; ++j) {
                int kv_id = invlists[i * max_cluster_size + j];
                memcpy(parse_value + (start_idx + j) * dim, values + kv_id * dim, dim * sizeof(uint16_t));
                memcpy(parse_key + (start_idx + j) * dim, keys + kv_id * dim, dim * sizeof(uint16_t));
            }
        }
    }

    void construction_func(
        // int* cluster_size_ptr    // (groups, n_centroids)
    ) {
        int max_seq_len = 0;
        for (int row = 0; row < group_num; ++row) {
            int seq_len = parse_clusters(cluster_size_ptr + row * n_centroids, row);
            last_seq_lengths[current_batch * group_num + row] = seq_len;
            if (max_seq_len == 0) max_seq_len = seq_len;
            else AT_ASSERT(max_seq_len == seq_len, "sequence length in one group is not equal.");
        }
        AT_ASSERT(static_cast<int64_t>(max_seq_len) <= input_seq_length, "Sum of the cluster sizes is larger than input sequence length.");
        
        pool_->LockQueue();
        for (int i = 0; i < group_num; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->organize_kv(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(group_num);
        pool_->UnlockQueue();
        pool_->NotifyAll();
    }

    // organize the keys & values by clustering results
    void async_construction(
        torch::Tensor& clusters,        // key id of each clusters, (groups, n_centroids, max_cluster_size), cpu int tensor
        torch::Tensor& cluster_size,    // cluster size of all clusters list, (groups, n_centroids), cpu int tensor
        int batch_idx
    ) {
        clusters_ptr = static_cast<int*>(clusters.data_ptr<int>());
        cluster_size_ptr = static_cast<int*>(cluster_size.data_ptr<int>());
        max_cluster_size = clusters.size(2);

        current_batch = batch_idx;

        // const int input_groups = clusters.size(0);
        // AT_ASSERT(input_groups == group_num, "Wrong group num of the vectors.");
        // const int input_n_centroids = clusters.size(1);
        // AT_ASSERT(input_n_centroids == n_centroids, "Wrong number of clusters.");

        // submit parse job
        pool_->LockQueue();
        pool_->QueueJobWOLock([this](void* para) { return this->construction_func(); }, nullptr);
        pool_->AddNumTask(1);
        pool_->UnlockQueue();
        pool_->NotifyAll();

        return;
    }

    void construction_sync() {
        // wait for construction finish
        pool_->Wait();

        if (current_batch == batch_size - 1) {
            // free the memory
            if (ivf_list_size_csum_array != nullptr) {
                delete[] ivf_list_size_csum_array;
                ivf_list_size_csum_array = nullptr;
            }
        }
        return;
    }



    // organize part of keys & values by clustering results
    void update_kv_func(
        void* para
        // uint16* input_key_ptr,      // (batch_groups, update_seq_len, dim)
        // uint16* input_value_ptr,    // (batch_groups, update_seq_len, dim)
        // int* clusters_ptr,          // (batch_groups, update_n_centroids, max_cluster_size)
        // int* cluster_size_ptr       // (batch_groups, update_n_centroids)
    ) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        for (int idx = _start; idx < _end; ++idx) {
            auto cluster_descriptors_group = cluster_descriptors + idx * static_cast<int64_t>(final_n_centroids) + last_n_centroids;

            int* invlists = clusters_ptr + idx * n_centroids * max_cluster_size;
            int* invlists_size = cluster_size_ptr + idx * n_centroids;

            uint16_t* values = input_value_ptr + idx * input_seq_length * dim;
            uint16_t* keys = input_key_ptr + idx * input_seq_length * dim;
            uint16_t* parse_key = ivf_key_array + idx * output_seq_length * dim;
            uint16_t* parse_value = ivf_value_array + idx * output_seq_length * dim;
            
            // parse clusters
            int start_idx = last_seq_lengths[idx];
            for (int i = 0; i < n_centroids; ++i) {
                int list_size = invlists_size[i];
                // calculate the number of blocks for this cluster
                const int block_num = (list_size + block_size - 1) / block_size;

                // initialize the cluster descriptor
                cluster_descriptors_group[i].inBlockCache = false;
                if (block_num > PRE_ALLOCATED_NUM) {
                    delete[] cluster_descriptors_group[i].GPUBlockIDs;
                    cluster_descriptors_group[i].GPUBlockIDs = new int[block_num];  // allocate larger memory
                }
                cluster_descriptors_group[i].CPUStartIndex = start_idx;
                cluster_descriptors_group[i].BlockNum = block_num;
                cluster_descriptors_group[i].LastBlockSize = list_size % block_size == 0 ? block_size : list_size % block_size;

                // organize keys & values by clusters
                for (int j = 0; j < list_size; ++j) {
                    int kv_id = invlists[i * max_cluster_size + j];
                    memcpy(parse_value + (start_idx + j) * dim, values + kv_id * dim, dim * sizeof(uint16_t));
                    memcpy(parse_key + (start_idx + j) * dim, keys + kv_id * dim, dim * sizeof(uint16_t));
                }

                start_idx += list_size;
            }
            last_seq_lengths[idx] = start_idx;
        }
    }

    // parallel organize batch_groups keys & values by clustering results
    void update_kv(
        torch::Tensor& update_keys,     // (batch_groups, update_seq_len, dim)
        torch::Tensor& update_values,   // (batch_groups, update_seq_len, dim)
        torch::Tensor& clusters,        // key id of each clusters, (batch_groups, update_n_centroids, max_cluster_size), cpu int tensor
        torch::Tensor& cluster_size     // cluster size of all clusters list, (batch_groups, update_n_centroids), cpu int tensor
    ) {
        if (update_keys.dtype() == torch::kFloat16) {     // fp16
            input_key_ptr = reinterpret_cast<uint16_t*>(update_keys.data_ptr<at::Half>());
            input_value_ptr = reinterpret_cast<uint16_t*>(update_values.data_ptr<at::Half>());
        } else {    // bf16
            input_key_ptr = reinterpret_cast<uint16_t*>(update_keys.data_ptr<at::BFloat16>());
            input_value_ptr = reinterpret_cast<uint16_t*>(update_values.data_ptr<at::BFloat16>());
        }
        input_seq_length = update_keys.size(1);

        clusters_ptr = static_cast<int*>(clusters.data_ptr<int>());
        cluster_size_ptr = static_cast<int*>(cluster_size.data_ptr<int>());
        n_centroids = clusters.size(1);
        max_cluster_size = clusters.size(2);

        // submit update tasks
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->update_kv_func(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
        pool_->Wait();  // sync

        last_n_centroids += n_centroids;
    }



    void batch_access(void* para) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        for (int i = _start; i < _end; ++i) {
            // used for hit keys
            auto hit_block_ids_group = hit_block_ids + i * buffer_size;
            auto hit_block_sizes_group = hit_block_sizes + i * buffer_size;
            auto hit_block_sizes_cumsum_group = hit_block_sizes_cumsum + i * buffer_size;
            // used for missing keys
            auto miss_block_ids_group = miss_block_ids + i * buffer_size;
            auto miss_block_sizes_group = miss_block_sizes + i * buffer_size;
            auto miss_block_sizes_cumsum_group = miss_block_sizes_cumsum + i * buffer_size;
            // access buffer manager
            auto [hit_num, miss_num, hit_block_num, miss_block_num] = caches[i]->batch_access(searched_clusters_ptr + i * nprobe, nprobe, 
                                                                                              hit_block_ids_group,
                                                                                              hit_block_sizes_group, 
                                                                                              hit_block_sizes_cumsum_group,
                                                                                              miss_block_ids_group,
                                                                                              miss_block_sizes_group,
                                                                                              miss_block_sizes_cumsum_group);
            // std::fill(hit_block_ids_group + hit_block_num, hit_block_ids_group + buffer_size, -1);
            // std::fill(hit_block_sizes_group + hit_block_num, hit_block_sizes_group + buffer_size, 0);
            // std::fill(hit_block_sizes_cumsum_group + hit_block_num, hit_block_sizes_cumsum_group + buffer_size, hit_block_sizes_cumsum_group[hit_block_num - 1]);
            // std::fill(miss_block_ids_group + miss_block_num, miss_block_ids_group + buffer_size, -1);
            // std::fill(miss_block_sizes_group + miss_block_num, miss_block_sizes_group + buffer_size, 0);
            // std::fill(miss_block_sizes_cumsum_group + miss_block_num, miss_block_sizes_cumsum_group + buffer_size, miss_block_sizes_cumsum_group[miss_block_num - 1]);

            hit_block_nums[i] = hit_block_num;
            miss_block_nums[i] = miss_block_num;
        }
    }

    void batch_update(void* para) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        for (int i = _start; i < _end; ++i) {
            auto update_cache_block_ids_group = update_cache_indices + i * buffer_size;
            auto update_block_sizes_group = update_block_sizes + i * buffer_size;
            auto update_buffer_indices_group = update_buffer_indices + i * buffer_size;
            auto [admiss_num, update_block_num] = caches[i]->batch_update(update_cache_block_ids_group,
                                                                          update_block_sizes_group,
                                                                          update_buffer_indices_group);
            // std::fill(update_cache_block_ids_group + update_block_num, update_cache_block_ids_group + buffer_size, -1);
            // std::fill(update_block_sizes_group + update_block_num, update_block_sizes_group + buffer_size, 0);
            // std::fill(update_buffer_indices_group + update_block_num, update_buffer_indices_group + buffer_size, 0);

            update_block_nums[i] = update_block_num;
        }
    }

    void para_batch_updata() {
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->batch_update(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
    }

    void para_batch_access() {
        // create tasks
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->batch_access(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
        pool_->Wait();

        // submit aysnc update tasks
        pool_->LockQueue();
        pool_->QueueJobWOLock([this](void* para) { return this->para_batch_updata(); }, nullptr);
        pool_->AddNumTask(1);
        pool_->UnlockQueue();
        pool_->NotifyAll();
        
        return;
    }

    void sync() {
        pool_->Wait();
        return;
    }

};


namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<WaveBufferCPU>(m, "WaveBufferCPU")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, MyThreadPool*>(),
             py::arg("batch_size"), py::arg("group_num"), py::arg("dim"), py::arg("nprobe"), py::arg("block_size"), 
             py::arg("n_centroids"), py::arg("final_n_centroids"), py::arg("buffer_size"), py::arg("capacity"), 
             py::arg("threads"), py::arg("pool"))
        .def("set_indices", &WaveBufferCPU::set_indices, 
            py::arg("hit_block_ids"), py::arg("hit_block_sizes"), py::arg("hit_block_sizes_cumsum"), py::arg("hit_block_nums"),
            py::arg("miss_block_ids"), py::arg("miss_block_sizes"), py::arg("miss_block_sizes_cumsum"), py::arg("miss_block_nums"),
            py::arg("update_buffer_indices"), py::arg("update_block_sizes"), py::arg("update_cache_indices"), py::arg("update_block_nums"), 
            py::arg("searched_clusters"))
        .def("set_kv", &WaveBufferCPU::set_kv, 
            py::arg("ivf_key"), py::arg("ivf_value"), py::arg("input_keys"), py::arg("input_values"))
        .def("async_construction", &WaveBufferCPU::async_construction, 
            py::arg("clusters"), py::arg("cluster_size"), py::arg("batch_idx"))
        .def("construction_sync", &WaveBufferCPU::construction_sync)
        .def("update_kv", &WaveBufferCPU::update_kv, 
            py::arg("update_keys"), py::arg("update_values"), py::arg("clusters"), py::arg("cluster_size"))
        .def("batch_access", &WaveBufferCPU::para_batch_access)
        .def("sync", &WaveBufferCPU::sync);
    
    py::class_<MyThreadPool>(m, "MyThreadPool")
        .def(py::init<>());
    
    py::class_<ThreadPool>(m, "ThreadPool")
        .def(py::init<int>())
        .def("get", &ThreadPool::get, py::return_value_policy::reference);
}
