#pragma once

#include <cuda_runtime.h>

// #include "ll/utils.cuh"
#include "ll/weights_storage.cuh"
#include "ll/mbyv.cuh"

namespace act {

// ReLU
__device__ __forceinline__ float relu(float x)
{
    return x > 0.f ? x : 0.f;
}

// SiLU (a.k.a. swish): x * sigmoid(x)
// This is optional and a bit heavier than ReLU; useful as a "bonus" experiment later.
__device__ __forceinline__ float silu(float x)
{
    // fast-ish sigmoid
    float s = 1.f / (1.f + __expf(-x));
    return x * s;
}

}  // namespace act

namespace ll {

struct ActIdentity {
    __device__ __forceinline__ float operator()(float x) const
    {
        return x;
    }
};

struct ActRelu {
    __device__ __forceinline__ float operator()(float x) const
    {
        return act::relu(x);
    }
};

// Dense layer kernel: y = act(W*x + b)
//
// W is stored in row-major with column padding to PADDED_COLS = WARPS * ceil_div(COLS, WARPS).
// Layout: W[row * PADDED_COLS + col]
//
// Kernel launch requirements:
//   gridDim.x  == BLOCKS
//   blockDim.x == 32
//   blockDim.y == WARPS
//
// If GLOB_N > 0, scratch must point to an array of
//   BLOCKS * WARPS * 32 * GLOB_N floats.
// If GLOB_N == 0, scratch can be nullptr.

template <size_t ROWS, size_t COLS, size_t BLOCKS, size_t WARPS, size_t REG_N, size_t SHMEM_N, size_t GLOB_N, class Act>
__global__ void dense_mbyv_kernel(const float *__restrict__ W_padded, const float *__restrict__ b,
                                  const float *__restrict__ x, float *__restrict__ y, float *__restrict__ scratch,
                                  Act act)
{
    static_assert(BLOCKS >= 1, "BLOCKS must be >= 1");
    static_assert(WARPS >= 1 && WARPS <= 32, "WARPS must be in [1,32]");

    // Provide at least 1 element to keep zero-sized shared arrays legal.
    __shared__ float shmem[(SHMEM_N > 0 ? SHMEM_N : 1) * WARPS * 32];

    float *sh_ptr = nullptr;
    if constexpr (SHMEM_N > 0) {
        sh_ptr = &shmem[(threadIdx.y * 32 + threadIdx.x) * SHMEM_N];
    }

    float *g_ptr = nullptr;
    if constexpr (GLOB_N > 0) {
        // scratch is indexed as: [block][warp][lane][g]
        const size_t tid_linear = ((size_t)blockIdx.x * WARPS + (size_t)threadIdx.y) * 32 + (size_t)threadIdx.x;
        g_ptr = &scratch[tid_linear * GLOB_N];
    }

    auto ws = make_weights_storage<REG_N, SHMEM_N, GLOB_N>(sh_ptr, g_ptr);
    using WS = decltype(ws);

    using M = MByV<WS, ROWS, COLS, BLOCKS, WARPS>;
    M m(ws);

    m.load_kernel(W_padded, b);

    float feats[M::FEATS_CNT_PER_THREAD];
    m.load_feats(x, feats);

    constexpr size_t ACC_N = M::BIG_ROW_ITER_CNT_PER_THREAD + M::P2_TASKS_CNT + M::task_split.cbi_cnt;
    float acc[ACC_N];
    m.calc(acc, feats);

    m.save_result_act(acc, y, act);
}

// Helper for padded column count
// PADDED_COLS = WARPS * ceil_div(COLS, WARPS)
template <size_t COLS, size_t WARPS>
struct PaddedCols {
    static constexpr size_t cols_per_warp = ceil_div(COLS, WARPS);
    static constexpr size_t padded_cols = WARPS * cols_per_warp;
};

}  // namespace ll
