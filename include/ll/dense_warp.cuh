#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ll {

// Simple activations for fused epilogue.
struct ActIdentity {
    __device__ __forceinline__ float operator()(float x) const
    {
        return x;
    }
};

struct ActRelu {
    __device__ __forceinline__ float operator()(float x) const
    {
        return x > 0.0f ? x : 0.0f;
    }
};

// Warp-per-output-row GEMV (batch=1) for row-major weights.
// Each warp computes one output neuron: y[row] = act(dot(W[row, :], x) + b[row]).
// We stage x into shared memory per block to reuse across warps.
//
// This kernel is intentionally "boring" but very competitive for tiny MLPs:
// - no per-inference weight staging (unlike the current MByV path)
// - coalesced weight loads (row-major, contiguous per row)
// - warp reductions via shfl
//
// BLOCK_WARPS: number of warps per block (e.g., 4 or 8)
template <int M, int K, int BLOCK_WARPS, class Act>
__global__ void dense_warp_rm_kernel(const float *__restrict__ W_rm,  // [M x K] row-major
                                     const float *__restrict__ b,     // [M]
                                     const float *__restrict__ x,     // [K]
                                     float *__restrict__ y,           // [M]
                                     Act act)
{
    static_assert(BLOCK_WARPS >= 1, "BLOCK_WARPS must be >= 1");
    constexpr int WARP = 32;

    // thread organization
    const int lane = threadIdx.x & (WARP - 1);
    const int warp_in_block = threadIdx.x >> 5;  // 0..BLOCK_WARPS-1
    const int warp_global = (int)blockIdx.x * BLOCK_WARPS + warp_in_block;
    const int row = warp_global;
    if (row >= M)
        return;

    // stage x in shared memory (reused by all warps)
    extern __shared__ float sx[];  // K floats
    // cooperative load: all threads in block participate
    for (int i = threadIdx.x; i < K; i += BLOCK_WARPS * WARP) {
        sx[i] = x[i];
    }
    __syncthreads();

    const float *wrow = W_rm + (size_t)row * (size_t)K;

    float acc = 0.0f;
    // Unroll by 4 for better ILP.
    // Each lane handles elements lane, lane+32, lane+64, ...
    for (int j = lane; j < K; j += WARP * 4) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int j0 = j;
        int j1 = j + WARP;
        int j2 = j + 2 * WARP;
        int j3 = j + 3 * WARP;
        if (j0 < K)
            s0 = wrow[j0] * sx[j0];
        if (j1 < K)
            s1 = wrow[j1] * sx[j1];
        if (j2 < K)
            s2 = wrow[j2] * sx[j2];
        if (j3 < K)
            s3 = wrow[j3] * sx[j3];
        acc += (s0 + s1) + (s2 + s3);
    }

    // warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        float v = acc + b[row];
        y[row] = act(v);
    }
}

}  // namespace ll
