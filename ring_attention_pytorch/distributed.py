# simple wrappers to avoid importing heavy deps when torch.distributed not initialized
import torch.distributed as dist

def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1 