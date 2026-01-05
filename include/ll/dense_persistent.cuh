#pragma once
#include <cuda_runtime.h>

namespace ll {

struct ActRelu {
    __device__ __forceinline__ float operator()(float x) const
    {
        return x > 0.0f ? x : 0.0f;
    }
};
struct ActIdentity {
    __device__ __forceinline__ float operator()(float x) const
    {
        return x;
    }
};

__device__ __forceinline__ float warp_reduce_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Computes y[m] = act( dot(W[m,:], x[:]) + b[m] )
// W is row-major [M x K], contiguous.
template <int M, int K, typename Act>
__device__ __forceinline__ void dense_tiled_warp_rm(const float *__restrict__ W, const float *__restrict__ b,
                                                    const float *__restrict__ x, float *__restrict__ y, Act act,
                                                    int warps_per_block)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;  // warp id within block

    for (int base = 0; base < M; base += warps_per_block) {
        int m = base + warp;
        float sum = 0.0f;
        if (m < M) {
            const float *row = W + m * K;
            for (int k = lane; k < K; k += 32)
                sum += row[k] * x[k];
            sum = warp_reduce_sum(sum);
            if (lane == 0)
                y[m] = act(sum + b[m]);
        }
        __syncthreads();
    }
}

}  // namespace ll
