import os, json, math
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def add_config_args(parser):
    parser.add_argument("--attn_type", type=str, default="RetroInfer",
                        choices=["Full_Flash_Attn", "RetroInfer"], help="Attention method")
    parser.add_argument("--retrieval_budget", type=float, default=0.018, help="Retrieval budget")
    parser.add_argument("--estimation_budget", type=float, default=0.232, help="Estimation budget for RetroInfer")
    parser.add_argument("--cache_ratio", type=float, default=0.0, help="Cache ratio for RetroInfer")
    parser.add_argument("--use_cuda_graph", action='store_true', help="Use CUDA graph for inference")
    parser.add_argument("--gpu_only", action='store_true', help="Whether to use GPU-only mode for RetroInfer")
    return parser


def get_numa_node_core_count(node_id=0):
    path = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
    if not path.exists():
        count = os.cpu_count()
        print(f"NUMA node{node_id} not found, set core to #total_cpu_core: {count}")
        return max(count - 2, 1)    # reserve 2 cores for system
    # get NUMA node core count
    cpulist = path.read_text().strip()
    count = 0
    for part in cpulist.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            count += end - start + 1
        else:
            count += 1
    return max(count - 2, 1)  # reserve 2 cores for system


def generate_config(
    model_name, context_len, attn_type, 
    retrieval_budget=0.018, estimation_budget=0.232, cache_ratio=0.0,
    use_cuda_graph=False, gpu_only=False
):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        _config = json.load(f)
    
    avg_cluster_size = 16
    n_segments = max(round(context_len/8192), 1)
    
    # compute the nearest multiple of lcm(8, n_segments) due to the kernel limitation
    n_factor = math.lcm(8, n_segments)
    n_clusters = max(round(context_len/avg_cluster_size), n_factor)
    lower = (n_clusters // n_factor) * n_factor
    upper = lower + n_factor
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper

    if attn_type == 'RetroInfer':
        _config[attn_type]['core'] = get_numa_node_core_count(0)
        _config[attn_type]['n_centroids'] = n_clusters
        _config[attn_type]['n_segment'] = n_segments
        _config[attn_type]['pages_per_cluster'] = round(avg_cluster_size / 8) # default page size is 8 vectors
        _config[attn_type]['retrieval_budget'] = retrieval_budget
        _config[attn_type]['estimation_budget'] = estimation_budget
        _config[attn_type]['cache_ratio'] = cache_ratio
        if context_len <= 4096: # increase buffer size for small context
            _config[attn_type]['buffer_cluster_num'] = 150
        _config[attn_type]['use_cuda_graph'] = use_cuda_graph
        _config[attn_type]['gpu_only'] = gpu_only
    
    if attn_type != "Full_Flash_Attn":
        print(_config[attn_type])
    
    return _config