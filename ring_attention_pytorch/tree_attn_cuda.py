import os
import math
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load
import pybind11
import flash_attn

# Directory setup -----------------------------------------------------------
_CUDA_SRC_DIR = Path(__file__).parent / "csrc"
_sources = [
    str(_CUDA_SRC_DIR / "tree_attn_cuda.cpp"),
    str(_CUDA_SRC_DIR / "tree_attn_cuda_kernel.cu"),
]

FA_MOD_ROOT = Path(__file__).parent / "third_party" / "flash_attn_mod"
FA_HDR3     = FA_MOD_ROOT / "csrc" / "flash_attn" / "src"

# derive header locations from flash_attn wheel (if installed)
flash_root = Path(flash_attn.__file__).parent
flash_hdr1 = flash_root / "src"
flash_hdr2 = flash_root / "flash_attn_core" / "include"

extra_include_paths = [
    pybind11.get_include(),
    str(flash_root), str(flash_hdr1), str(flash_hdr2),
    str(FA_MOD_ROOT / "csrc"), str(FA_MOD_ROOT / "src"),
    str(FA_MOD_ROOT / "flash_attn_core" / "include"),
    str(FA_HDR3),
]
# remove extra _sources entry; kernels come via headers

# Build / load extension (lazy) -------------------------------------------

_extension = None

def _load_ext():
    global _extension
    if _extension is None:
        _extension = load(
            name="tree_attn_cuda",
            sources=_sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            extra_include_paths=extra_include_paths,
            verbose=False,
        )
    return _extension

# Public API ----------------------------------------------------------------

def tree_attn_decode_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, fused: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA kernel implementation of tree attention decoding (local part).

    This variant *only* replaces the local attention computation and LSE
    calculation.  The caller is still responsible for performing the three
    distributed all-reduce operations exactly as in the original
    `tree_attn_decode` function.

    Inputs must be float32 CUDA tensors with shapes:
      q : (B, H, 1, D)
      k : (B, H, N, D)
      v : (B, H, N, D)
    """
    ext = _load_ext()
    if fused:
        # placeholder: fused kernel not wired yet
        print("[tree_attn_cuda] fused=True – fused kernel not yet implemented, falling back")
    out, lse = ext.forward(q, k, v)
    if out.dtype != q.dtype:
        out = out.to(q.dtype)
    return out, lse 