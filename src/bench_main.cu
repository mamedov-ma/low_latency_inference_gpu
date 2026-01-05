
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "bench_utils.h"

// A "do-nothing" kernel to estimate launch overhead and jitter.
__global__ void empty_kernel()
{
    // prevent full elimination
    asm volatile("");
}

int main()
{
    print_device_info();

    cudaStream_t stream {};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Bench empty kernel launch (event overhead can be comparable to very small kernels).
    // We report two numbers:
    //  - inner_reps=1   : simplest measurement
    //  - inner_reps=32  : amortized measurement (often more stable)
    for (int inner_reps : {1, 32}) {
        auto empty_times = bench_stream_us([&](cudaStream_t s) { empty_kernel<<<1, 1, 0, s>>>(); }, stream,
                                           /*warmup=*/2000,
                                           /*iters=*/20000,
                                           /*inner_reps=*/inner_reps);
        CUDA_CHECK(cudaGetLastError());
        auto empty_stats = compute_percentiles(std::move(empty_times));
        print_stats(std::string("empty_kernel_launch (inner_reps=") + std::to_string(inner_reps) + ")", empty_stats);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
