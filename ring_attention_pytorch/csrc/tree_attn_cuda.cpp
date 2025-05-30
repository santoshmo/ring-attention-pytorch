// C++ / PyBind11 binding for Tree Attention CUDA kernel

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// Declarations from the CUDA file
void launch_tree_attn_fwd(
    const void* q,
    const void* k,
    const void* v,
    float* out,
    float* lse,
    int B, int H, int N, int D,
    float scale,
    cudaDataType_t dtype,
    cudaStream_t stream);

// Thin wrapper that checks input tensors and dispatches to CUDA kernel

std::vector<torch::Tensor> tree_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), "dtype mismatch");
    TORCH_CHECK(q.dtype() == torch::kFloat32 || q.dtype() == torch::kHalf, "supports fp32 or fp16");

    auto B = q.size(0);
    auto H = q.size(1);
    auto D = q.size(3); // q shape (B, H, 1, D)
    auto N = k.size(2);

    auto out = torch::zeros({B, H, 1, D}, q.options().dtype(torch::kFloat32));
    auto lse = torch::zeros({B, H, 1}, q.options().dtype(torch::kFloat32));

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    cudaStream_t stream = 0;
    if (q.dtype() == torch::kFloat32) {
        launch_tree_attn_fwd(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            lse.data_ptr<float>(),
            B, H, N, D, scale, CUDA_R_32F, stream);
    } else {
        launch_tree_attn_fwd(
            q.data_ptr<at::Half>(),
            k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(),
            out.data_ptr<float>(),
            lse.data_ptr<float>(),
            B, H, N, D, scale, CUDA_R_16F, stream);
    }

    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &tree_attn_forward, "Tree Attention forward (CUDA)");
} 