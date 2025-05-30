// Phase 3 – Tensor-Core Fused Tree Attention – Stage 1
// Implements tiled cp.async loading of Q / K slices (no compute yet).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <mma.h>

// Compile-time tile sizes (override via -D flags)
#ifndef FUSED_Q_TILE
#define FUSED_Q_TILE 16
#endif
#ifndef FUSED_K_TILE
#define FUSED_K_TILE 16
#endif
#ifndef FUSED_D_TILE
#define FUSED_D_TILE 64
#endif

// cp.async macro (same as Phase-2)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define HAS_CP_ASYNC 1
#else
#define HAS_CP_ASYNC 0
#endif

#if HAS_CP_ASYNC
#define CP_ASYNC_G2S(dst, src)                                                        \
    do {                                                                              \
        uint32_t smem_addr;                                                           \
        asm ("cvta.to.shared.u32 %0, %1;" : "=r"(smem_addr) : "l"(dst));          \
        asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" ::                 \
                     "r"(smem_addr), "l"(src), "n"(16));                          \
    } while(0)
#else
#define CP_ASYNC_G2S(dst, src) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src)
#endif

// Kernel – one block per (B, H) pair handling 16 queries

template<int D>
__global__ void tree_attn_fwd_kernel_fused_stage1(
    const __half* __restrict__ q,   // (B,H,T_q,D) – we assume T_q multiple of 16
    const __half* __restrict__ k,   // (B,H,T_k,D)
    int B, int H, int T_q, int T_k)
{
    extern __shared__ __align__(16) __half sm[];
    // Layout: first Q tile (FUSED_Q_TILE × D), then K tile (FUSED_K_TILE × D)
    __half* q_sh = sm;
    __half* k_sh = q_sh + FUSED_Q_TILE * D;

    const int bh = blockIdx.x;
    const int b  = bh / H;
    const int h  = bh % H;
    const int q_base = blockIdx.y * FUSED_Q_TILE; // 16-query micro-batch index
    if(q_base >= T_q) return;

    // Pointer bases
    const __half* q_ptr = q + ((b * H + h) * T_q + q_base) * D;
    const __half* k_ptr = k + ((b * H + h) * T_k) * D; // start of key sequence

    // ------------------------------------------------------------------
    // Load Q tile to shared (synchronous for now)
    for(int tid = threadIdx.x; tid < FUSED_Q_TILE * D; tid += blockDim.x)
        q_sh[tid] = q_ptr[tid];

    // Load first K tile (indices 0..15) to shared
    for(int tid = threadIdx.x; tid < FUSED_K_TILE * D; tid += blockDim.x)
        k_sh[tid] = k_ptr[tid];

#if HAS_CP_ASYNC
    __syncthreads();
#endif

    // No compute yet – stage 1 only validates data movement

#if USE_TENSOR_OPS && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda;
    // Use first warp to compute 16x16 dot products of Q (16xD) and K (D x16)
    if(threadIdx.x < 32){
        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> fragB;
        wmma::fragment<wmma::accumulator, 16,16,16, float> fragC;
        wmma::fill_fragment(fragC, 0.0f);

        // For D=64 we need 4 iterations (16 columns each)
        const int iterations = D / 16;
        for(int it=0; it<iterations; ++it){
            const __half* a_tile = q_sh + it*16;               // row-major
            const __half* b_tile = k_sh + it*16;               // col-major
            wmma::load_matrix_sync(fragA, a_tile, D);
            wmma::load_matrix_sync(fragB, b_tile, D);
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        // store logits to shared memory (FP32)
        float* logits_sh = reinterpret_cast<float*>(k_sh + FUSED_K_TILE * D);
        wmma::store_matrix_sync(logits_sh, fragC, 16, wmma::mem_row_major);
    }
#endif
}

// Host launcher wrapper used only for developmental testing (not wired yet)
void launch_tree_attn_fwd_fused_stage1(
    const void* q, const void* k,
    int B, int H, int T_q, int T_k, int D, cudaStream_t stream)
{
    dim3 grid(B * H, (T_q + FUSED_Q_TILE - 1) / FUSED_Q_TILE);
    dim3 block(128);
    size_t shmem = sizeof(__half) * (FUSED_Q_TILE + FUSED_K_TILE) * D;
    tree_attn_fwd_kernel_fused_stage1<FUSED_D_TILE><<<grid, block, shmem, stream>>>(
        static_cast<const __half*>(q), static_cast<const __half*>(k), B, H, T_q, T_k);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Fused stage1 kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Keep original stub launcher so existing build continues to link
extern "C" __global__ void tree_attn_fwd_kernel_fused_stub();

void launch_tree_attn_fwd_fused(
    const void* q, const void* k, const void* v, float* out, float* lse,
    int B, int H, int N, int D, float scale, cudaDataType_t dtype, cudaStream_t stream)
{
    // Currently still a stub until full fused kernel arrives
    tree_attn_fwd_kernel_fused_stub<<<1,1,0,stream>>>();
} 