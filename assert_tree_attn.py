import torch
from torch import einsum
import torch.distributed as dist
import time
from contextlib import contextmanager
from collections import defaultdict

from ring_attention_pytorch import tree_attn_decode

"""
Tree Attention Profiling Test Script

This script tests tree attention decoding and provides detailed profiling 
of computation vs communication costs.

Usage examples:

1. Basic functionality test:
   python assert_tree_attn.py --world-size 4 --seq-len 1024

2. Enable profiling with CUDA:
   python assert_tree_attn.py --world-size 4 --seq-len 1024 --use-cuda --enable-profiling

3. Detailed profiling with more iterations:
   python assert_tree_attn.py --world-size 8 --seq-len 4096 --use-cuda --enable-profiling --profile-iterations 20

The profiling output will show:
- Computation times: local attention, sharding, numerator/denominator calculation
- Communication times: max LSE reduction, denominator sum, numerator sum  
- Summary breakdown of computation vs communication percentage

Requirements: torch, click, einops
"""

# regular attention for testing

def regular_decode(q, k, v):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    attn = sim.softmax(dim = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# for testing the above tree decoding function
# `pip install click` as requirement, besides `torch`

import os
import click
from math import ceil

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(
    rank,
    world_size,
    use_cuda
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = "gloo" if not use_cuda else "nccl"
    dist.init_process_group(backend, rank = rank, world_size = world_size)

    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size,
    dim,
    heads,
    batch,
    seq_len,
    use_cuda,
    enable_profiling,
    profile_iterations,
):
    setup(rank, world_size, use_cuda)
    is_main = rank == 0

    ring_seq_size = ceil(seq_len / world_size)

    # inputs

    q = torch.randn(batch, heads, 1, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)

    if use_cuda:
        q, k, v = tuple(t.cuda(rank).half() for t in (q, k, v))

    # easy forcing all q, k, v to be same across all device

    dist.all_reduce(q)
    dist.all_reduce(k)
    dist.all_reduce(v)

    # outputs

    out = regular_decode(q, k, v)
    
    if enable_profiling:
        # Initialize timer
        timer = Timer(use_cuda=use_cuda)
        
        # Warmup runs (don't time these)
        if is_main:
            print(f"Running {profile_iterations} warmup iterations...")
        
        for _ in range(min(3, profile_iterations)):
            tree_out = profiled_tree_attn_decode(q, k, v, Timer(use_cuda), warmup=True)
        
        # Synchronize all processes before starting timing
        dist.barrier()
        
        if is_main:
            print(f"Running {profile_iterations} profiled iterations...")
        
        # Timed runs
        for i in range(profile_iterations):
            tree_out = profiled_tree_attn_decode(q, k, v, timer)
            
        # Synchronize before printing results
        dist.barrier()
        
        # Print profiling results
        if is_main:
            timer.print_stats(rank=rank)
            
        # Also print per-rank stats for distributed insights
        dist.barrier()
        for r in range(world_size):
            if rank == r:
                if not is_main:  # Don't print main rank twice
                    timer.print_stats(rank=rank)
            dist.barrier()
            
    else:
        tree_out = tree_attn_decode(q, k, v)

    out = out.to(tree_out.dtype)

    # if not main early return

    if not is_main:
        return cleanup()

    # if is main, validate output is the same for kv sequence split across machines vs without

    tree_out = tree_out.cpu()
    out = out.cpu()

    output_atol = 1e-2 if use_cuda else 1e-5

    assert torch.allclose(tree_out, out, atol = output_atol), '🟥 output is not the same'

    print('✅ output is the same between tree and non-tree attention decoding')

    cleanup()

@click.command()
@click.option('--world-size', default = 8, help = 'number of machines / processes')
@click.option('--dim', default = 64, help = 'dimension')
@click.option('--heads', default = 8, help = 'dimension')
@click.option('--batch', default = 1, help = 'dimension')
@click.option('--use-cuda', is_flag = True, help = 'whether to test with CUDA and NCCL')
@click.option('--seq-len', default = 31, help = 'sequence length to test')
@click.option('--enable-profiling', is_flag = True, help = 'enable detailed profiling of computation and communication')
@click.option('--profile-iterations', default = 10, help = 'number of iterations to run for profiling statistics')
def test(
    world_size: int,
    dim: int,
    heads: int,
    batch: int,
    use_cuda: bool,
    seq_len: int,
    enable_profiling: bool,
    profile_iterations: int,
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (world_size, dim, heads, batch, seq_len, use_cuda, enable_profiling, profile_iterations),
        nprocs = world_size,
        join = True
    )

# Profiling utilities

class Timer:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda
        self.times = defaultdict(list)
        
    @contextmanager
    def time_block(self, name):
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        else:
            start_time = time.perf_counter()
            
        try:
            yield
        finally:
            if self.use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)  # milliseconds
            else:
                elapsed = (time.perf_counter() - start_time) * 1000  # convert to milliseconds
                
            self.times[name].append(elapsed)
    
    def get_stats(self):
        stats = {}
        for name, times in self.times.items():
            stats[name] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times),
                'count': len(times)
            }
        return stats
    
    def print_stats(self, rank=None):
        stats = self.get_stats()
        prefix = f"[Rank {rank}] " if rank is not None else ""
        print(f"\n{prefix}Profiling Results (all times in ms):")
        print("=" * 60)
        
        # Separate computation and communication
        comp_times = {k: v for k, v in stats.items() if 'comm_' not in k}
        comm_times = {k: v for k, v in stats.items() if 'comm_' in k}
        
        if comp_times:
            print("COMPUTATION:")
            for name, stat in comp_times.items():
                print(f"  {name:25s}: {stat['mean']:8.3f} ± {stat['max']-stat['min']:6.3f} (total: {stat['total']:8.3f})")
        
        if comm_times:
            print("COMMUNICATION:")
            for name, stat in comm_times.items():
                print(f"  {name:25s}: {stat['mean']:8.3f} ± {stat['max']-stat['min']:6.3f} (total: {stat['total']:8.3f})")
        
        if comp_times and comm_times:
            total_comp = sum(stat['total'] for stat in comp_times.values())
            total_comm = sum(stat['total'] for stat in comm_times.values())
            total_time = total_comp + total_comm
            print(f"\nSUMMARY:")
            print(f"  Total Computation  : {total_comp:8.3f} ms ({total_comp/total_time*100:5.1f}%)")
            print(f"  Total Communication: {total_comm:8.3f} ms ({total_comm/total_time*100:5.1f}%)")
            print(f"  Total Time         : {total_time:8.3f} ms")

def profiled_tree_attn_decode(q, k, v, timer, warmup=False):
    """Tree attention decode with detailed profiling"""
    
    # Import here to access the internals for profiling
    from ring_attention_pytorch.tree_attn_decoding import get_rank, get_world_size
    from einops import rearrange
    import torch.distributed as dist
    
    with timer.time_block("total_tree_attn"):
        q_prec_dims, dtype = q.shape[:-1], q.dtype
        eps = 1e-8
        shard_kv_seq = True
        use_triton = None if not q.is_cuda else True
        
        # Shard KV sequences
        with timer.time_block("comp_shard_kv"):
            if shard_kv_seq:
                dim_v = v.shape[-1]
                rank, world_size = get_rank(), get_world_size()
                k = k.chunk(world_size, dim = -2)
                v = v.chunk(world_size, dim = -2)
                k, v = (k[rank], v[rank]) if rank < len(k) else (None, None)

        if k is not None:
            # Local attention computation
            with timer.time_block("comp_local_attention"):
                if use_triton and q.is_cuda:
                    from ring_attention_pytorch.triton_flash_attn import flash_attn_forward
                    local_out, _, lse = flash_attn_forward(
                        q, k, v,
                        causal = False,
                        return_normalized_output = True,
                        load_accumulated = False,
                        head_first_dim = True,
                        remove_padding = True
                    )
                    lse = rearrange(lse, '... -> ... 1')
                else:
                    scale = q.shape[-1] ** -0.5
                    sim = einsum('... i d, ... j d -> ... i j', q, k) * scale
                    lse = sim.logsumexp(dim = -1, keepdim = True)
                    attn = sim.softmax(dim = -1)
                    local_out = einsum('... i j, ... j d -> ... i d', attn, v)
        else:
            # Handle edge case
            with timer.time_block("comp_edge_case"):
                local_out = q.new_zeros((*q_prec_dims, dim_v), dtype = torch.float32)
                lse = torch.full((*q_prec_dims, 1), -torch.finfo(torch.float32).max, device = q.device, dtype = torch.float32)

        # Communication phase 1: find max LSE
        with timer.time_block("comm_max_lse"):
            max_lse = lse.clone()
            dist.all_reduce(max_lse, dist.ReduceOp.MAX)

        # Compute numerator and denominator
        with timer.time_block("comp_num_den"):
            den = (lse - max_lse).exp()
            num = local_out * den

        # Communication phase 2: sum denominator
        with timer.time_block("comm_sum_den"):
            dist.all_reduce(den)
            
        # Communication phase 3: sum numerator  
        with timer.time_block("comm_sum_num"):
            dist.all_reduce(num)

        # Final computation
        with timer.time_block("comp_final_output"):
            out = num.div_(den.clamp(min = eps))
            out = out.type(dtype)

    return out

if __name__ == '__main__':
    test()
