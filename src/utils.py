import torch
import gc

def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
