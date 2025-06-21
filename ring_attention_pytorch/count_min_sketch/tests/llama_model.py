#!/usr/bin/env python
"""llama_model.py

A minimal implementation to load and run Meta's original Llama format.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


class LlamaModel(torch.nn.Module):
    """A minimal implementation to load and run Meta's original Llama format."""
    
    def __init__(self, model_path: str | Path):
        super().__init__()
        self.model_path = Path(model_path)
        
        # Load model config
        with open(self.model_path / "params.json") as f:
            self.config = json.load(f)

        self.dim = self.config["dim"]
        self.n_layers = self.config["n_layers"]
        self.n_heads = self.config["n_heads"]
        self.n_kv_heads = self.config.get("n_kv_heads", self.n_heads) # Defaults to n_heads if not GQA/MQA
        
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"Number of query heads ({self.n_heads}) must be divisible by "
                f"number of key/value heads ({self.n_kv_heads})."
            )
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads
        
        # Corrected output dimension for K and V projections if GQA/MQA is used
        kv_projection_output_dim = self.n_kv_heads * self.head_dim
        
        # Load model weights
        ckpt = torch.load(self.model_path / "consolidated.00.pth", map_location="cpu")
        self.weights = ckpt
        
        # Initialize layers
        self.embed_tokens = torch.nn.Embedding(self.config["vocab_size"], self.dim)
        self.embed_tokens.weight.data = self.weights["tok_embeddings.weight"]
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            layer = torch.nn.ModuleDict({
                "self_attn": torch.nn.ModuleDict({
                    "q_proj": torch.nn.Linear(self.dim, self.dim, bias=False), # n_heads * head_dim
                    "k_proj": torch.nn.Linear(self.dim, kv_projection_output_dim, bias=False), # n_kv_heads * head_dim
                    "v_proj": torch.nn.Linear(self.dim, kv_projection_output_dim, bias=False), # n_kv_heads * head_dim
                    "o_proj": torch.nn.Linear(self.dim, self.dim, bias=False),
                }),
                "mlp": torch.nn.ModuleDict({
                    "gate_proj": torch.nn.Linear(self.dim, self.config["multiple_of"] * 4, bias=False),
                    "up_proj": torch.nn.Linear(self.dim, self.config["multiple_of"] * 4, bias=False),
                    "down_proj": torch.nn.Linear(self.config["multiple_of"] * 4, self.dim, bias=False),
                }),
                "input_layernorm": torch.nn.LayerNorm(self.dim, eps=self.config.get("norm_eps", 1e-5)),
                "post_attention_layernorm": torch.nn.LayerNorm(self.dim, eps=self.config.get("norm_eps", 1e-5)),
            })
            
            # Load weights
            prefix = f"layers.{i}."
            layer["self_attn"]["q_proj"].weight.data = self.weights[f"{prefix}attention.wq.weight"]
            layer["self_attn"]["k_proj"].weight.data = self.weights[f"{prefix}attention.wk.weight"]
            layer["self_attn"]["v_proj"].weight.data = self.weights[f"{prefix}attention.wv.weight"]
            layer["self_attn"]["o_proj"].weight.data = self.weights[f"{prefix}attention.wo.weight"]
            layer["mlp"]["gate_proj"].weight.data = self.weights[f"{prefix}feed_forward.w1.weight"]
            layer["mlp"]["up_proj"].weight.data = self.weights[f"{prefix}feed_forward.w3.weight"]
            layer["mlp"]["down_proj"].weight.data = self.weights[f"{prefix}feed_forward.w2.weight"]
            layer["input_layernorm"].weight.data = self.weights[f"{prefix}attention_norm.weight"]
            # Handle potential missing bias for layernorms if model saved without it
            if f"{prefix}attention_norm.bias" in self.weights:
                layer["input_layernorm"].bias.data = self.weights[f"{prefix}attention_norm.bias"]
            layer["post_attention_layernorm"].weight.data = self.weights[f"{prefix}ffn_norm.weight"]
            if f"{prefix}ffn_norm.bias" in self.weights:
                layer["post_attention_layernorm"].bias.data = self.weights[f"{prefix}ffn_norm.bias"]
            
            self.layers.append(layer)
        
        self.norm = torch.nn.LayerNorm(self.dim, eps=self.config.get("norm_eps", 1e-5))
        self.norm.weight.data = self.weights["norm.weight"]
        if "norm.bias" in self.weights:
             self.norm.bias.data = self.weights["norm.bias"]

        # Store hidden states for analysis
        self.hidden_states = []
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that also stores hidden states."""
        batch_size, seq_len = input_ids.shape
        
        self.hidden_states = []  # Reset hidden states
        
        # Get embeddings
        hidden_states_current_layer = self.embed_tokens(input_ids)
        self.hidden_states.append(hidden_states_current_layer.detach().clone())
        
        for layer_idx, layer_module in enumerate(self.layers):
            # ===== DEVICE CHECK FOR LAYER PARAMETERS =====
            # Assuming hidden_states_current_layer is on the correct device for reference
            expected_param_device = hidden_states_current_layer.device 
            for name, param in layer_module.named_parameters():
                if param.device != expected_param_device:
                    print(f"WARNING: Layer {layer_idx}, Param '{name}' is on {param.device} but expected {expected_param_device}!")
            # =============================================

            residual = hidden_states_current_layer
            normed_hidden_state = layer_module["input_layernorm"](hidden_states_current_layer)
            # print(f"Layer {layer_idx}: normed_hidden_state.device: {normed_hidden_state.device}") # DEBUG DEVICE

            # print(f'Layer {layer_idx}: k_proj.weight.device: {layer_module["self_attn"]["k_proj"].weight.device}') # DEBUG DEVICE
            # print(f'Layer {layer_idx}: v_proj.weight.device: {layer_module["self_attn"]["v_proj"].weight.device}') # DEBUG DEVICE

            q_raw = layer_module["self_attn"]["q_proj"](normed_hidden_state)
            k_raw = layer_module["self_attn"]["k_proj"](normed_hidden_state)
            v_raw = layer_module["self_attn"]["v_proj"](normed_hidden_state)
            # print(f"Layer {layer_idx}: q_raw.device: {q_raw.device}, k_raw.device: {k_raw.device}, v_raw.device: {v_raw.device}") # DEBUG DEVICE

            q = q_raw.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_raw.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_raw.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

            # Ensure contiguity and correct device
            q = q.contiguous().to(normed_hidden_state.device) # Ensuring q is also explicitly on device 
            k = k.contiguous().to(normed_hidden_state.device) # Using normed_hidden_state.device for consistency
            v = v.contiguous().to(normed_hidden_state.device)

            # Temporarily remove sdp_kernel context for debugging device issue
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            # Force k and v to the same device as q, before the print and call
            target_device = q.device # q is confirmed on cuda:0
            if k.device != target_device:
                print(f"Layer {layer_idx}: MOVING k from {k.device} to {target_device} BEFORE SDP CALL (sdp_kernel_removed)") # DEBUG
                k = k.to(target_device)
            if v.device != target_device:
                print(f"Layer {layer_idx}: MOVING v from {v.device} to {target_device} BEFORE SDP CALL (sdp_kernel_removed)") # DEBUG
                v = v.to(target_device)

            # Explicitly create copies with .to(copy=True) and re-cast dtype right before the call
            current_q_dtype = q.dtype # Assuming q has the correct dtype
            # Ensure q itself is also a fresh copy on the right device/dtype for consistency
            q_final = q.to(device=target_device, dtype=current_q_dtype, copy=True)
            k_final = k.to(device=target_device, dtype=current_q_dtype, copy=True)
            v_final = v.to(device=target_device, dtype=current_q_dtype, copy=True)

            print(f"Layer {layer_idx}: Pre-SDP (sdp_kernel_removed, after .to(copy=True)): q_final.device={q_final.device}, q_final.dtype={q_final.dtype}, k_final.device={k_final.device}, k_final.dtype={k_final.dtype}, v_final.device={v_final.device}, v_final.dtype={v_final.dtype}") 
            attn_output_reshaped = F.scaled_dot_product_attention(q_final, k_final, v_final, is_causal=True)

            attn_output = attn_output_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            
            hidden_states_out_attn = layer_module["self_attn"]["o_proj"](attn_output)
            hidden_states_current_layer = residual + hidden_states_out_attn
            
            # MLP
            residual = hidden_states_current_layer
            normed_hidden_state_mlp = layer_module["post_attention_layernorm"](hidden_states_current_layer)
            
            gate = layer_module["mlp"]["gate_proj"](normed_hidden_state_mlp)
            up = layer_module["mlp"]["up_proj"](normed_hidden_state_mlp)
            hidden_states_mlp_activated = F.silu(gate) * up
            hidden_states_out_mlp = layer_module["mlp"]["down_proj"](hidden_states_mlp_activated)
            hidden_states_current_layer = residual + hidden_states_out_mlp
            
            self.hidden_states.append(hidden_states_current_layer.detach().clone()) # Clone to be safe
        
        # Final norm
        hidden_states_final_norm = self.norm(hidden_states_current_layer)
        self.hidden_states.append(hidden_states_final_norm.detach().clone()) # Clone to be safe
        
        return hidden_states_final_norm

