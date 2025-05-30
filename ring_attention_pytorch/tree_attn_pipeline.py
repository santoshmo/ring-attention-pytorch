# coding: utf-8
"""
Tree-Attention pipeline helper that overlaps NCCL communication of query
q-1 with FlashAttention-2 compute of query q.

Key methods
===========
step(q,k,v)   : feeds a single (B,H,1,D) query row. Returns the *previous*
                query's fully reduced output whenever it becomes available
                (first call returns None).
finalize()    : flushes the last in-flight query and returns its output.
run_chunk(Q,K,V): utility that accepts (B,H,T_q,D) with T_q>1 and processes
                  each row through step/finalize, returning a contiguous
                  tensor of outputs (same shape as Q).

The helper relies on:
• FlashAttention-2 for local soft-max (via tree_attn_cuda.tree_attn_decode_cuda).
• The hierarchical + mixed-precision all-reduce helpers already defined in
  tree_attn_decoding.py (_hierarchical_all_reduce).

Only single-row decode is latency-critical; this class gives overlap there
and also accelerates small-T_q pre-fill via run_chunk().
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import torch

from ring_attention_pytorch.tree_attn_cuda import tree_attn_decode_cuda
from ring_attention_pytorch.tree_attn_decoding import _hierarchical_all_reduce

EPS = 1e-8


class TreeAttnPipeline:
    """Overlap FlashAttention compute with NCCL all-reduce (ring pipeline)."""

    def __init__(self, max_inflight: int = 2):
        assert max_inflight >= 1, "max_inflight must be >=1"
        self.comm_stream = torch.cuda.Stream()
        self.max_inflight = max_inflight
        self.pending: List[Tuple[torch.cuda.Event, torch.Tensor]] = []

        # staging for the *current* query (which will be sent next step)
        self._prev_lse: Optional[torch.Tensor] = None  # fp32 (B,H,1)
        self._prev_local_out: Optional[torch.Tensor] = None  # fp32 (B,H,1,D)
        self._prev_dtype: Optional[torch.dtype] = None

    # ------------------------------------------------------------------
    def _launch_comm_prev(self):
        """All-reduce previous query's tensors on comm stream and enqueue event."""
        if self._prev_lse is None:
            return  # nothing to send (first step)

        with torch.cuda.stream(self.comm_stream):
            # 1) global max reduction
            max_lse = self._prev_lse.clone()
            _hierarchical_all_reduce(max_lse, op=torch.distributed.ReduceOp.MAX)

            # 2) compute denominator & numerator in fp32
            den = (self._prev_lse - max_lse).exp()  # fp32
            num = self._prev_local_out.to(torch.float32) * den  # fp32

            # 3) cast numerator to bf16 to cut bandwidth
            num_bf16 = num.to(torch.bfloat16)

            # 4) hierarchical all-reduce
            _hierarchical_all_reduce(den)        # fp32 tiny
            _hierarchical_all_reduce(num_bf16)   # bf16 large

            # 5) final normalisation
            num_fp32 = num_bf16.to(torch.float32)
            out = num_fp32.div(den.clamp(min=EPS))
            if self._prev_dtype != torch.float32:
                out = out.to(self._prev_dtype)

        evt = torch.cuda.Event()
        evt.record(self.comm_stream)
        self.pending.append((evt, out))

        # clear staging
        self._prev_lse = None
        self._prev_local_out = None
        self._prev_dtype = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Optional[torch.Tensor]:
        """Process one (B,H,1,D) query row.

        Returns the fully reduced output of the *previous* row whenever
        it is ready. First call returns None (warm-up latency).
        """
        # flush if queue depth reached
        if len(self.pending) >= self.max_inflight:
            evt, _ = self.pending.pop(0)
            evt.synchronize()

        # schedule comm for previous query (overlap)
        self._launch_comm_prev()

        # compute current query (default stream)
        local_out, lse = tree_attn_decode_cuda(q, k, v, fused=True)

        # stage for next round
        self._prev_lse = lse.contiguous()
        self._prev_local_out = local_out.contiguous()
        self._prev_dtype = local_out.dtype

        # return earliest completed result if any
        if self.pending and self.pending[0][0].query():
            evt, out_ready = self.pending.pop(0)
            return out_ready
        return None

    # ------------------------------------------------------------------
    def finalize(self) -> torch.Tensor:
        """Flush pipeline and return final query's output."""
        self._launch_comm_prev()

        out_final: Optional[torch.Tensor] = None
        while self.pending:
            evt, out = self.pending.pop(0)
            evt.synchronize()
            out_final = out
        if out_final is None:
            raise RuntimeError("finalize() called without any step()")
        return out_final

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_chunk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Process a chunk with shape (B,H,T_q,D), return same-shaped output."""
        assert q.ndim == 4 and q.shape[2] > 0, "q must be (B,H,T_q,D) with T_q>0"
        outs = []
        for t in range(q.shape[2]):
            ready = self.step(q[:, :, t:t+1, :], k, v)
            if ready is not None:
                outs.append(ready)
        outs.append(self.finalize())
        return torch.cat(outs, dim=2) 