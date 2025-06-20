#!/usr/bin/env python
"""llama_layer_recovery.py

Compute a CountSketch heavy-hitter recovery score for each transformer layer of a
locally downloaded Llama3.2-1B (or compatible) model in Meta's original format.

The script loads the model, runs it on a short prompt, obtains every layer's
hidden states, and evaluates how well a CountSketch with modest width can
recover the k largest-magnitude hidden-dimension entries ("heavy hitters").

Usage (from repository root with your virtualenv active):

    python -m ring_attention_pytorch.count_min_sketch.llama_layer_recovery \
        --model_path /path/to/llama3.2-1B \
        --prompt "Hello world" \
        --k 16 --m 256 --seed 42

The output prints per-layer overlap counts and returns a JSON-serialisable
summary dictionary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import os
import sys

# Ensure project root (two levels up) is on PYTHONPATH when run directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)  # allow 'ring_attention_pytorch' absolute imports

# Import CountSketch from sibling file (same directory)
try:
    from .test_flash_tree_sketch import CountSketch  # type: ignore
    from .llama_model import LlamaModel
except ImportError:  # Fallback when executed as a standalone script
    from ring_attention_pytorch.count_min_sketch.test_flash_tree_sketch import CountSketch  # type: ignore
    from ring_attention_pytorch.count_min_sketch.llama_model import LlamaModel

def evaluate_model_layer_recovery(
    model_path: str | Path,
    prompt: str = "Hello world",
    k: int = 16,
    m: int = 256,
    seed: int = 0,
    rows: int = 8,
    dtype: torch.dtype | None = torch.float16,
    device: torch.device | None = None,
) -> List[Dict[str, Any]]:
    """Return list of per-layer recovery dictionaries.

    Each dict contains:
      {"layer": int, "overlap": int, "score": float}
    where *score* = overlap / k.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # DEBUG

    # ------------------------------------------------------------------
    # Load tokenizer and model (support both Meta original format and HF)
    # ------------------------------------------------------------------
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model: torch.nn.Module
    original_format_ok = False
    try:
        # Attempt to load Meta original-format weights first (params.json etc.)
        model = LlamaModel(model_path)  # type: ignore
        original_format_ok = True
    except (FileNotFoundError, OSError):
        # Fallback: load Hugging Face-format checkpoint
        print("Falling back to Hugging Face transformers model loading …")
        hf_kwargs = {}
        # If a dtype is specified, pass it directly to avoid large fp32 weights on CPU
        if dtype is not None:
            hf_kwargs["torch_dtype"] = dtype
        hf_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)  # type: ignore

    # Move model to device / dtype where possible (AutoModel may already have correct dtype)
    model = model.to(device=device, dtype=dtype or torch.float16)
    print(f"Model device after .to(): {next(model.parameters()).device}")  # DEBUG
    model.eval()

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    print(f"Input_ids device: {input_ids.device}") # DEBUG

    # --------------------------------------------------------------
    # Forward pass and capture hidden states (works for both models)
    # --------------------------------------------------------------
    with torch.no_grad():
        try:
            # Hugging Face models expose hidden_states when requested
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = list(outputs.hidden_states)  # type: ignore
        except TypeError:
            # Our custom LlamaModel does not accept those kwargs
            _ = model(input_ids)
            hidden_states = model.hidden_states  # type: ignore

    # Determine hidden dimension d from first hidden state tensor
    # shape: (batch, seq_len, hidden_dim)
    d = hidden_states[1].shape[-1]
    print(f"Model hidden dimension d = {d}")

    # Initialise one CountSketch object to reuse across layers
    sketcher = CountSketch(d=d, m=m, seed=seed, num_rows=rows)

    layer_scores: List[Dict[str, Any]] = []

    for layer_idx, layer_h in enumerate(hidden_states[1:], start=1):
        # Average across batch and sequence length to obtain a single vector of
        # length d representing this layer's activation profile.
        vec = layer_h.mean(dim=(0, 1))  # shape (d,)

        # ------------------------------------------------------------
        # 1) CMS build + heavy-hitter identification
        # ------------------------------------------------------------
        sketch = sketcher.encode(vec)
        candidate_k = min(4 * k, d)  # safety margin: keep 4×k candidates (<= d)
        candidate_set = set(sketcher.heavy_hitter_indices(sketch, candidate_k).tolist())

        # Ground-truth top-k for overlap metric (kept for comparison)
        true_topk = set(torch.topk(vec.abs(), k).indices.cpu().tolist())
        overlap = len(true_topk.intersection(candidate_set))

        # ------------------------------------------------------------
        # 2) Exact-value fetch (trivial here because we already own `vec`)
        # ------------------------------------------------------------
        candidate_list = sorted(candidate_set)
        vec32 = vec.float()
        exact_heavy_vals = vec32[candidate_list]
        total_mass = vec32.sum()
        heavy_sum = exact_heavy_vals.sum()
        residual = total_mass - heavy_sum

        # Reconstruct vector from heavy values + residual in dummy slot 0
        recon = torch.zeros_like(vec32)  # float32 tensor
        recon[candidate_list] = exact_heavy_vals
        if 0 not in candidate_set:
            recon[0] += residual

        diff = vec32 - recon
        l2_error = torch.norm(diff).item()
        max_abs_err = torch.max(diff.abs()).item()
        mass_error = (heavy_sum + residual - total_mass).abs().item()
        print(f"Δmass = {mass_error:.3e}  (tolerance = {(1e-6 + 1e-5*total_mass.abs()).item():.3e})")
        mass_check = torch.allclose(exact_heavy_vals.sum() + residual, total_mass, rtol=1e-5, atol=1e-6)

        print(
            f"Layer {layer_idx:02d} |H|={len(candidate_list):3d} | "
            f"overlap {overlap}/{k} | L2 err {l2_error:.2e} | "
            f"max err {max_abs_err:.2e} | mass ok: {'✔' if mass_check else '✘'}"
        )

        layer_scores.append(
            {
                "layer": layer_idx,
                "overlap": overlap,
                "score": overlap / k,
                "l2_error": l2_error,
                "max_abs_error": max_abs_err,
                "mass_check": mass_check,
                "candidate_size": len(candidate_list),
            }
        )

    return layer_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CountSketch per-layer recovery for a Llama3.2-1B model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--prompt", default="Hello world", help="Prompt to feed into the model")
    parser.add_argument("--k", type=int, default=16, help="Number of heavy hitters to recover per layer")
    parser.add_argument("--m", type=int, default=512, help="Sketch width (number of buckets)")
    parser.add_argument("--rows", type=int, default=8, help="Number of CMS rows (hash functions)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for hashing")
    parser.add_argument(
        "--output_json", type=Path, default=None, help="Optional path to write JSON summary"
    )
    args = parser.parse_args()

    scores = evaluate_model_layer_recovery(
        model_path=args.model_path,
        prompt=args.prompt,
        k=args.k,
        m=args.m,
        seed=args.seed,
        rows=args.rows,
    )

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Per-layer recovery scores written to {args.output_json}")


if __name__ == "__main__":
    main() 