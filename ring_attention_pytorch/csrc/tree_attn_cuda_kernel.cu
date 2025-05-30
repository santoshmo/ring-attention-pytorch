// Copyright 2024
// Tree Attention Decoding – CUDA Forward Kernel
//
// NOTE: This implementation is a *reference* kernel focused on clarity rather
// than ultimate performance.  It supports only the forward pass required for
// decoding (no gradients) and assumes FP32 inputs/outputs.
//
// Input shapes (row-major / contiguous):
//   q : (B, H, D)
//   k : (B, H, N, D)
//   v : (B, H, N, D)
//   All dimensions use the same stride pattern as a contiguous PyTorch tensor.
//
// Output shapes:
//   out : (B, H, D)
//   lse : (B, H, 1)
//
// The kernel launches one thread-block per (batch, head) pair and processes
// the whole key sequence serially inside the block.  For typical transformer
// dimensions (D ≤ 128, N ≤ 8192), this design yields reasonable performance
// while remaining simple.
//
// For production use, consider a more sophisticated fused kernel similar to
// FlashAttention or migrate to Triton.

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

// After includes add
#ifndef USE_TENSOR_OPS
#define USE_TENSOR_OPS 0
#endif

// Helper routine - will later be swapped to tensor-core mma.sync when USE_TENSOR_OPS==1
#if USE_TENSOR_OPS && defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=800)
#include <mma.h>
using namespace nvcuda;
#endif

template<int D>
__device__ inline float dot_qk_naive(const float * __restrict__ q, const __half* __restrict__ k){
    float acc = 0.f;
#pragma unroll
    for(int i=0;i<D;++i)
        acc += q[i]*__half2float(k[i]);
    return acc;
}

template<int D>
__device__ inline float dot_qk(const float* __restrict__ q, const __half* __restrict__ k){
#if USE_TENSOR_OPS && defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=800)
    // TODO: implement mma.sync path. For now, fall back to naive.
#endif
    return dot_qk_naive<D>(q,k);
}

// Utilities ---------------------------------------------------------------

#ifndef CHECK_CUDA
#ifndef __CUDA_ARCH__
#define CHECK_CUDA(call)                                                     \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                   \
            printf("CUDA Error %s at %s:%d\n", cudaGetErrorString(err),      \
                   __FILE__, __LINE__);                                      \
        }                                                                   \
    } while (0)
#else
#define CHECK_CUDA(call) call
#endif
#endif

// Forward kernel ----------------------------------------------------------

template<typename scalar_t>
struct traits;

template<>
struct traits<float> {
    using scalar = float;
    using acc_t  = float;
    static __device__ __forceinline__ float scale(float v, float s) { return v * s; }
};

template<>
struct traits<__half> {
    using scalar = __half;
    using acc_t  = float;
    static __device__ __forceinline__ float scale(__half v, float s) {
        return __half2float(v) * s;
    }
};

template<typename scalar_t, int MAX_D>
__global__ void tree_attn_fwd_kernel_t(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    float* __restrict__ out,
    float* __restrict__ lse,
    int B, int H, int N, int D,
    float scale)
{
    using acc_t = typename traits<scalar_t>::acc_t;

    int bh = blockIdx.x;
    int b  = bh / H;
    int h  = bh % H;

    if (b >= B) return;

    const scalar_t* q_ptr = q + (b * H + h) * D;               // (D)
    const scalar_t* k_ptr = k + ((b * H + h) * N) * D;         // (N, D)
    const scalar_t* v_ptr = v + ((b * H + h) * N) * D;         // (N, D)

    float* out_ptr = out + (b * H + h) * D;                 // (D)
    float* lse_ptr = lse + (b * H + h);

    extern __shared__ float sm[];
    float* q_sh  = sm;
    float* out_sh = sm + D;

    // load q to shared (converted to float)
    for (int d = threadIdx.x; d < D; d += blockDim.x)
    {
        q_sh[d] = traits<scalar_t>::scale(q_ptr[d], scale);
    }
    __syncthreads();

#ifndef MAX_D_LOCAL
#define MAX_D_LOCAL 128
#endif
    acc_t local_max = -CUDART_INF_F;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const scalar_t* k_j = k_ptr + j * D;
        acc_t dot = 0.f;
        for (int d = 0; d < D; ++d)
            dot += q_sh[d] * traits<scalar_t>::scale(k_j[d], 1.f); // k already scaled
        local_max = fmaxf(local_max, dot);
    }

    // warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = local_max;
    __syncthreads();
    float max_logit = block_max;

    acc_t exp_sum = 0.f;
    float out_local[MAX_D_LOCAL];
    for (int d=0; d<D; ++d) out_local[d] = 0.f;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const scalar_t* k_j = k_ptr + j * D;
        const scalar_t* v_j = v_ptr + j * D;
        acc_t dot = 0.f;
        for (int d=0; d<D; ++d)
            dot += q_sh[d] * traits<scalar_t>::scale(k_j[d],1.f);
        float w = __expf(dot - max_logit);
        exp_sum += w;
        for (int d=0; d<D; ++d)
            out_local[d] += w * traits<scalar_t>::scale(v_j[d],1.f);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        exp_sum += __shfl_down_sync(0xffffffff, exp_sum, offset);

    __shared__ float block_sum_exp;
    if (threadIdx.x==0) block_sum_exp = exp_sum;
    __syncthreads();
    float denom = block_sum_exp + 1e-6f;

    // write per-thread accum to shared
    for (int d = 0; d < D; ++d)
        out_sh[threadIdx.x * D + d] = out_local[d];
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int d=0; d<D; ++d) {
            float sum = 0.f;
            for (int t=0; t<blockDim.x; ++t)
                sum += out_sh[t*D + d];
            out_ptr[d] = sum / denom;
        }
        *lse_ptr = logf(denom) + max_logit;
    }
}

// ---------------------------------------------------------------------------
// Phase-2 Optimized Kernel (tensor-core friendly, cp.async tiled)
// ---------------------------------------------------------------------------
// The template parameters below can be overridden at compile-time with
// -DBLOCK_SIZE=<threads> -DTILE_D=<d> -DTILE_N=<n> -DSTAGES=<k>
// to explore the design space without editing this file.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef TILE_D
#define TILE_D 64
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef STAGES
#define STAGES 2
#endif

// Helper: issue cp.async or fall back to regular global->shared copy
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_CP_ASYNC)
#define HAS_CP_ASYNC 1
#else
#define HAS_CP_ASYNC 0
#endif

#if HAS_CP_ASYNC
#define CP_ASYNC_GLOBAL_TO_SH(dst, src) {                                            \
        uint32_t smem_addr;                                                         \
        asm ("cvta.to.shared.u32 %0, %1;" : "=r"(smem_addr) : "l"(dst));       \
        asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" ::               \
                     "r"(smem_addr), "l"(src), "n"(16));                      \
    }
#else
#define CP_ASYNC_GLOBAL_TO_SH(dst, src) *reinterpret_cast<uint4*>(dst) =              \
        *reinterpret_cast<const uint4*>(src);
#endif

// Each block processes one (B,H) pair and TILE_N keys per iteration.
// Supports fp16 path only; fp32 falls back to reference kernel.

template<int TILE_N_, int TILE_D_, int STAGES_>
__global__ void tree_attn_fwd_kernel_opt_half(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ out,
    float* __restrict__ lse,
    int B, int H, int N, int D,
    float scale)
{
    static_assert(TILE_D_ % 16 == 0, "TILE_D must be multiple of 16");
    static_assert(TILE_N_ % 16 == 0, "TILE_N must be multiple of 16");

    // one block == one query vector
    const int bh = blockIdx.x;
    const int b  = bh / H;
    const int h  = bh % H;
    if (b >= B) return;

    // Pointers
    const __half* q_ptr = q + (b * H + h) * D;
    const __half* k_ptr = k + ((b * H + h) * N) * D;
    const __half* v_ptr = v + ((b * H + h) * N) * D;

    float* out_ptr = out + (b * H + h) * D;
    float* lse_ptr = lse + (b * H + h);

    // Registers – load q once (to FP32)
    float q_reg[TILE_D_];
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE)
        q_reg[d] = __half2float(q_ptr[d]) * scale;

    // Allocate shared memory tiles (double-buffered)
    extern __shared__ __align__(16) __half sm_opt[];
    __half* k_sh    = sm_opt;                                   // [STAGES_][TILE_N_][TILE_D_]
    __half* v_sh    = k_sh + STAGES_*TILE_N_*TILE_D_;

    // Local accumulators
    float max_logit = -CUDART_INF_F;
    float exp_sum   = 0.f;
    float out_local[TILE_D_];
    for(int d=0; d<D; ++d) out_local[d]=0.f;

    constexpr int LD = TILE_D_;

    // Pipeline over K/V tiles
    int tile_idx = 0;
    int stage    = 0;

    __shared__ float denom_acc_sh;
    __shared__ float out_red[TILE_D_];
    if(threadIdx.x < TILE_D_) out_red[threadIdx.x] = 0.f;
    if(threadIdx.x==0) denom_acc_sh = 0.f;

#if HAS_CP_ASYNC
    // Pre-fetch first stage
    for (int t = threadIdx.x * 16; t < TILE_N_*TILE_D_*sizeof(__half); t += BLOCK_SIZE*16) {
        const char* src = reinterpret_cast<const char*>(k_ptr) + t;
        char* dst       = reinterpret_cast<char*>(k_sh + stage*TILE_N_*TILE_D_) + t;
        CP_ASYNC_GLOBAL_TO_SH(dst, src);
        const char* srcv = reinterpret_cast<const char*>(v_ptr) + t;
        char* dstv      = reinterpret_cast<char*>(v_sh + stage*TILE_N_*TILE_D_) + t;
        CP_ASYNC_GLOBAL_TO_SH(dstv, srcv);
    }
#if HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();
#endif
#else
    // synchronous copy fallback
    for (int tn = threadIdx.x; tn < TILE_N_; tn += BLOCK_SIZE)
        for (int d = 0; d < D; ++d) {
            k_sh[tn*LD + d] = k_ptr[tn*D + d];
            v_sh[tn*LD + d] = v_ptr[tn*D + d];
        }
    __syncthreads();
#endif

    // Main loop over tiles
    while (tile_idx < N) {
        // Compute dot products for this tile
        for (int j = threadIdx.x; j < TILE_N_ && (tile_idx + j) < N; j += BLOCK_SIZE) {
            float dot = dot_qk<TILE_D_>(q_reg, &k_sh[stage*TILE_N_*LD + j*LD]);
            max_logit = fmaxf(max_logit, dot);
        }
        __syncthreads();

        // TODO: pipeline next tile with cp.async (double buffer)
        int next_tile_start = tile_idx + TILE_N_;
        int next_stage = (stage + 1) % STAGES_;
#if HAS_CP_ASYNC
        if (next_tile_start < N) {
            for (int t = threadIdx.x * 16; t < TILE_N_*TILE_D_*sizeof(__half); t += BLOCK_SIZE*16) {
                const char* src = reinterpret_cast<const char*>(k_ptr + next_tile_start*D) + t;
                char* dst       = reinterpret_cast<char*>(k_sh + next_stage*TILE_N_*TILE_D_) + t;
                CP_ASYNC_GLOBAL_TO_SH(dst, src);
                const char* srcv = reinterpret_cast<const char*>(v_ptr + next_tile_start*D) + t;
                char* dstv      = reinterpret_cast<char*>(v_sh + next_stage*TILE_N_*TILE_D_) + t;
                CP_ASYNC_GLOBAL_TO_SH(dstv, srcv);
            }
            asm volatile("cp.async.commit_group;\n"::);
        }
#endif
        __syncthreads();

        // Softmax accum
        float denom_block = 0.f;
        for (int j = threadIdx.x; j < TILE_N_ && (tile_idx + j) < N; j += BLOCK_SIZE) {
            float dot = dot_qk<TILE_D_>(q_reg, &k_sh[stage*TILE_N_*LD + j*LD]);
            float w = __expf(dot);
            denom_block += w;
            for (int d = 0; d < D; ++d)
                out_local[d] += w * __half2float(v_sh[stage*TILE_N_*LD + j*LD + d]);
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            denom_block += __shfl_down_sync(0xffffffff, denom_block, offset);
        }
        if ((threadIdx.x & 31) == 0) atomicAdd(&denom_acc_sh, denom_block);
        // accumulate out_local to shared
        for(int d=0; d<D; ++d)
            atomicAdd(&out_red[d], out_local[d]);
        __syncthreads();

        // Prepare for next tile
#if HAS_CP_ASYNC
        if (next_tile_start < N) asm volatile("cp.async.wait_all;\n"::);
#endif
        tile_idx += TILE_N_;
        stage = next_stage;
        __syncthreads();
    }

    // Final block-wide reductions
    // Max logit reduction across threads
    for (int offset = 16; offset > 0; offset >>= 1)
        max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));

    __shared__ float final_max;
    if(threadIdx.x==0) final_max = max_logit;
    __syncthreads();

    if(threadIdx.x < D){
        float val = out_red[threadIdx.x] / (denom_acc_sh + 1e-6f);
        out_ptr[threadIdx.x] = val;
    }
    if(threadIdx.x==0){
        *lse_ptr = logf(denom_acc_sh + 1e-6f) + final_max;
    }
}

// ---------------------------------------------------------------------------
// Launcher chooses optimized kernel for fp16, otherwise reference version.
// ---------------------------------------------------------------------------

void launch_tree_attn_fwd(
    const void* q,
    const void* k,
    const void* v,
    float* out,
    float* lse,
    int B, int H, int N, int D,
    float scale,
    cudaDataType_t dtype,
    cudaStream_t stream)
{
    dim3 grid(B * H);
    dim3 block(32);
    size_t shmem = sizeof(float) * D * (block.x + 1);

    if (dtype == CUDA_R_16F) {
        // launch optimized tensor-core path when D <= TILE_D and arch >= 800
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if (D <= TILE_D && block.x == 32) {
            dim3 block_opt(BLOCK_SIZE);
            size_t shmem_bytes = sizeof(__half)*STAGES*TILE_N*TILE_D*2;
            tree_attn_fwd_kernel_opt_half<TILE_N, TILE_D, STAGES><<<grid, block_opt, shmem_bytes, stream>>>(
                reinterpret_cast<const __half*>(q),
                reinterpret_cast<const __half*>(k),
                reinterpret_cast<const __half*>(v),
                out, lse, B, H, N, D, scale);
        } else
#endif
        {
            tree_attn_fwd_kernel_t<__half, 128><<<grid, block, shmem, stream>>>(
                reinterpret_cast<const __half*>(q),
                reinterpret_cast<const __half*>(k),
                reinterpret_cast<const __half*>(v),
                out, lse, B, H, N, D, scale);
        }
    } else {
        tree_attn_fwd_kernel_t<float, 128><<<grid, block, shmem, stream>>>(
            reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v),
            out, lse, B, H, N, D, scale);
    }

    CHECK_CUDA(cudaGetLastError());
} 