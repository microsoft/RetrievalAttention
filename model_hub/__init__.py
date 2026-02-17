
from .llama import LlamaModel
from .qwen import QwenModel
from transformers import AutoTokenizer


def add_model_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",
                                 "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                                 "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"], 
                        help="Huggingface model name")
    return parser


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name, max_len, dtype, device, tokenizer=None):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
                         max_length=max_len,
                         dtype=dtype,
                         device_map=device,
                         tokenizer=tokenizer)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
                        max_length=max_len,
                        dtype=dtype,
                        device_map=device,
                        tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm