import torch
import pytest

from ring_attention_pytorch.tree_attn_cuda import tree_attn_decode_cuda

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_attn_cuda_fp16_vs_fp32():
    B, H, D, N = 2, 4, 64, 256

    # random inputs
    q_fp32 = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    k_fp32 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v_fp32 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # fp32 reference kernel
    out32, _ = tree_attn_decode_cuda(q_fp32, k_fp32, v_fp32)

    # fp16 path
    q_fp16 = q_fp32.half()
    k_fp16 = k_fp32.half()
    v_fp16 = v_fp32.half()
    out16, _ = tree_attn_decode_cuda(q_fp16, k_fp16, v_fp16)

    # compare (cast fp16 output back to fp32)
    torch.testing.assert_close(out32, out16.to(torch.float32), rtol=1e-2, atol=1e-2) 