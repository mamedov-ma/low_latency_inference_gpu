#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "dense_warp.cuh"  // for ll::ActIdentity / ll::ActRelu

namespace ll {

// Thread-per-output-row GEMV with packed weights.
// Weights layout: W_cm has shape [K, Mpad] stored as W_cm[k*Mpad + row].
// Bias: b_pad[Mpad] (zeros in padded rows).
//
// Algorithm mapping (kept): each thread computes one output row;
// x is broadcast inside each warp via shuffles.
template <int M, int K, int Mpad, class Act>
__global__ void dense_mbyv_packed_kernel(const float *__restrict__ W_cm, const float *__restrict__ b_pad,
                                         const float *__restrict__ x, float *__restrict__ y, Act act)
{
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= M)
        return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned FULL_MASK = 0xffffffffu;

    // Each lane loads its own x values for each 32-wide segment.
    constexpr int SEG = (K + 31) / 32;
    float xseg[SEG];
#pragma unroll
    for (int s = 0; s < SEG; ++s) {
        const int k = s * 32 + lane;
        xseg[s] = (k < K) ? x[k] : 0.0f;
    }

    float acc = b_pad[row];

// For each k, fetch x_k from owner lane and multiply by weight for our row.
// Access is coalesced across the warp because W_cm is laid out as [k][row].
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int s = k >> 5;      // /32
        const int owner = k & 31;  // %32
        const float xk = __shfl_sync(FULL_MASK, xseg[s], owner);
        const float w = W_cm[k * Mpad + row];
        acc = fmaf(w, xk, acc);
    }

    y[row] = act(acc);
}

// Convenience launcher: chooses a simple 1D launch with enough threads for padded rows.
// Keeps warps full to preserve the intended intra-warp shuffle behaviour.
template <int M, int K, class Act>
inline void launch_dense_mbyv_packed(cudaStream_t stream, const float *W_cm, const float *b_pad, const float *x,
                                     float *y, Act act)
{
    constexpr int Mpad = (M + 31) / 32 * 32;
    constexpr int threads = Mpad;  // one thread per (padded) row
    static_assert(threads % 32 == 0, "threads must be multiple of 32");
    const dim3 block(threads, 1, 1);
    const dim3 grid(1, 1, 1);
    dense_mbyv_packed_kernel<M, K, Mpad, Act><<<grid, block, 0, stream>>>(W_cm, b_pad, x, y, act);
}

}  // namespace ll
