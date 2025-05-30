from __future__ import annotations

import torch
from torch import einsum, Tensor
import torch.distributed as dist

from einops import rearrange

from ring_attention_pytorch.distributed import get_rank, get_world_size

from ring_attention_pytorch.tensor_typing import Float

import socket

# -----------------------------------------------------------------------------
# Hierarchical collectives (intra-node then inter-node)
# -----------------------------------------------------------------------------

_LOCAL_GROUP = None
_LEADER_GROUP = None
_LOCAL_LEADER_RANK = None
_IS_LOCAL_LEADER = False


def _init_hierarchical_groups():
    global _LOCAL_GROUP, _LEADER_GROUP, _LOCAL_LEADER_RANK, _IS_LOCAL_LEADER
    if _LOCAL_GROUP is not None or not dist.is_initialized() or dist.get_world_size() == 1:
        return

    rank = dist.get_rank()
    world = dist.get_world_size()
    host = socket.gethostname()
    hosts = [None] * world
    dist.all_gather_object(hosts, host)

    host_to_ranks = {}
    for r, h in enumerate(hosts):
        host_to_ranks.setdefault(h, []).append(r)

    local_ranks = host_to_ranks[host]
    _LOCAL_GROUP = dist.new_group(ranks=local_ranks)
    _LOCAL_LEADER_RANK = min(local_ranks)
    _IS_LOCAL_LEADER = (rank == _LOCAL_LEADER_RANK)

    leader_ranks = sorted({min(ranks) for ranks in host_to_ranks.values()})
    if len(leader_ranks) > 1:
        _LEADER_GROUP = dist.new_group(ranks=leader_ranks)
    else:
        _LEADER_GROUP = None


def _hierarchical_all_reduce(t: torch.Tensor, op=dist.ReduceOp.SUM):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    _init_hierarchical_groups()
    dist.all_reduce(t, op=op, group=_LOCAL_GROUP)
    if _LEADER_GROUP is not None:
        if _IS_LOCAL_LEADER:
            dist.all_reduce(t, op=op, group=_LEADER_GROUP)
        dist.broadcast(t, src=_LOCAL_LEADER_RANK, group=_LOCAL_GROUP)
    return t

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main function

@torch.no_grad()
def tree_attn_decode(
    q: Float['b h 1 d'],
    k: Float['b h n d'] | None = None,
    v: Float['b h n dv'] | None = None,
    eps = 1e-8,
    shard_kv_seq = True,
    use_triton = None,
    use_cuda_kernel: bool = False
) -> Float['b h 1 dv']:

    q_prec_dims, dtype = q.shape[:-1], q.dtype

    assert not (exists(k) ^ exists(v)), 'keys and values are either both None, or both present'

    """
    Algorithm 3 proposed in Tree Attention
    https://arxiv.org/abs/2408.04093
    """

    # each machine (rank) takes care of a chunk of kv sequence within the world of many machines

    if shard_kv_seq:
        assert exists(k), 'keys and values must be passed if not already sharded across sequence'
        dim_v = v.shape[-1]

        rank, world_size = get_rank(), get_world_size()
        k = k.chunk(world_size, dim = -2)
        v = v.chunk(world_size, dim = -2)

        k, v = (k[rank], v[rank]) if rank < len(k) else (None, None)

    if exists(k):
        # calculate local output and lse

        # decide which implementation to use for the local attention
        use_triton = default(use_triton, q.is_cuda and not use_cuda_kernel)

        assert not (use_triton and not q.is_cuda), 'input needs to be on cuda if forcing the use of triton'

        if use_cuda_kernel:
            from ring_attention_pytorch.tree_attn_cuda import tree_attn_decode_cuda

            local_out, lse = tree_attn_decode_cuda(q, k, v)

        elif use_triton and q.is_cuda:
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
        # handle edge case where seq length < world size

        local_out = q.new_zeros((*q_prec_dims, dim_v), dtype = torch.float32)
        lse = torch.full((*q_prec_dims, 1), -torch.finfo(torch.float32).max, device = q.device, dtype = torch.float32)

    # first get global max(lse)
    max_lse = lse.clone()
    _hierarchical_all_reduce(max_lse, op=dist.ReduceOp.MAX)

    # derive numerator and denominator locally
    den = (lse - max_lse).exp()             # fp32 single element
    num = local_out * den                  # fp32 D elements

    # mixed-precision comms: bf16 numerator
    num_bf16 = num.to(torch.bfloat16)

    _hierarchical_all_reduce(den)
    _hierarchical_all_reduce(num_bf16)

    num = num_bf16.to(torch.float32)
    out = num.div_(den.clamp(min=eps))

    return out.type(dtype)