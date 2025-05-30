// Copyright 2024
// Tree Attention – CUDA Thin Wrapper around FlashAttention-2
// This file now only contains:
//   • calc_lse_kernel : tiny kernel to compute log-sum-exp per (B,H) row
//   • launch_tree_attn_fwd : wraps flash_attn_fwd and launches calc_lse_kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdio>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <flash_attn/flash.h>
#include <flash_attn/flash_api.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { cudaError_t err = call; if(err!=cudaSuccess){ fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); } } while(0)
#endif

// -------------------------------------------------------------
//  Kernel: compute row-wise log-sum-exp (B*H threads per block)
// -------------------------------------------------------------
__device__ inline void atomicMaxFloat(float* addr, float val){
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i, assumed;
    int new_val;
    do{
        assumed = old;
        float old_f = __int_as_float(assumed);
        float max_f = fmaxf(old_f, val);
        new_val = __float_as_int(max_f);
        old = atomicCAS(addr_i, assumed, new_val);
    }while(assumed!=old);
}

__global__ void calc_lse_kernel(const __half* __restrict__ q,
                                const __half* __restrict__ k,
                                float* __restrict__ lse,
                                int B,int H,int N,int D,float scale)
{
    extern __shared__ float q_sh[]; // D floats
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    const __half* q_ptr = q + (b*H + h)*D;
    const __half* k_ptr = k + ((b*H + h)*N)*D;

    // load q to shared (FP32 scaled)
    for(int d=threadIdx.x; d<D; d+=blockDim.x)
        q_sh[d] = __half2float(q_ptr[d])*scale;
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for(int n=threadIdx.x; n<N; n+=blockDim.x){
        const __half* k_row = k_ptr + n*D;
        float dot=0.f;
        for(int d=0; d<D; ++d)
            dot += q_sh[d]*__half2float(k_row[d]);
        local_max = fmaxf(local_max, dot);
    }
    // warp reduce max
    for(int off=16; off>0; off>>=1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    __shared__ float row_max;
    if((threadIdx.x & 31)==0) atomicMaxFloat(&row_max, local_max);
    __syncthreads();

    float max_logit = row_max;

    float local_sum = 0.f;
    for(int n=threadIdx.x; n<N; n+=blockDim.x){
        const __half* k_row = k_ptr + n*D;
        float dot=0.f;
        for(int d=0; d<D; ++d)
            dot += q_sh[d]*__half2float(k_row[d]);
        local_sum += __expf(dot - max_logit);
    }
    for(int off=16; off>0; off>>=1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    __shared__ float row_denom;
    if((threadIdx.x & 31)==0) atomicAdd(&row_denom, local_sum);
    __syncthreads();

    if(threadIdx.x==0)
        lse[bh] = logf(row_denom + 1e-6f) + max_logit;
}

// -------------------------------------------------------------
//  Host launcher wrapping FlashAttention-2 forward pass
// -------------------------------------------------------------
void launch_tree_attn_fwd(const void* q,const void* k,const void* v,
                          float* out,float* lse,
                          int B,int H,int N,int D,float scale,
                          cudaDataType_t dtype,cudaStream_t stream)
{
    TORCH_CHECK(dtype == CUDA_R_16F, "Current wrapper expects fp16 inputs");

    // build at::Tensor wrappers for flash_attn_fwd API
    auto opts_h = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
    at::Tensor q_t = at::from_blob(const_cast<void*>(q), {B,1,H,D}, opts_h);
    at::Tensor k_t = at::from_blob(const_cast<void*>(k), {B,N,H,D}, opts_h);
    at::Tensor v_t = at::from_blob(const_cast<void*>(v), {B,N,H,D}, opts_h);

    at::Tensor out_t = at::from_blob(out, {B,1,H,D}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    flash_attn_fwd(out_t, q_t, k_t, v_t,
                   /*p_dropout=*/0.f,
                   /*is_causal=*/false,
                   /*softmax_scale=*/scale,
                   stream);

    dim3 grid(B*H);
    dim3 block(64);
    size_t shm = sizeof(float)*D;
    calc_lse_kernel<<<grid, block, shm, stream>>>(static_cast<const __half*>(q),
                                                 static_cast<const __half*>(k),
                                                 lse, B,H,N,D, scale);
    CHECK_CUDA(cudaGetLastError());
} 