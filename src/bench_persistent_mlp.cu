#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <cstring>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_persistent.cuh"

using namespace ll;

struct alignas(64) TickBuffer {
    volatile uint32_t seq_in;
    volatile uint32_t seq_out;
    volatile uint32_t stop;
    uint32_t pad;
    float x[ll::cfg::IN];
    float y[ll::cfg::OUT];
};

// very light backoff to reduce CPU burn in polling loops
static inline void cpu_pause()
{
#if defined(__x86_64__) || defined(_M_X64)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

__global__ void persistent_mlp_kernel(TickBuffer *tb, const float *__restrict__ W1, const float *__restrict__ b1,
                                      const float *__restrict__ W2, const float *__restrict__ b2,
                                      const float *__restrict__ W3, const float *__restrict__ b3)
{
    using namespace ll::cfg;
    __shared__ float xsh[IN];
    __shared__ float h1[H1];
    __shared__ float h2[H2];

    uint32_t last = 0;

    const int tid = threadIdx.x;

    while (true) {
        if (tb->stop)
            return;

        // Spin until a new sequence arrives
        uint32_t s = tb->seq_in;
        if (s == last) {
            __nanosleep(200);
            continue;
        }
        last = s;

        // load input from mapped host memory into shared
        for (int i = tid; i < IN; i += blockDim.x)
            xsh[i] = tb->x[i];
        __syncthreads();

        // layer1: IN -> H1 (ReLU)
        ll::dense_tiled_warp_rm<H1, IN>(W1, b1, xsh, h1, ll::ActRelu {}, WARPS_PER_BLOCK);

        // layer2: H1 -> H2 (ReLU)
        ll::dense_tiled_warp_rm<H2, H1>(W2, b2, h1, h2, ll::ActRelu {}, WARPS_PER_BLOCK);

        // layer3: H2 -> OUT (Identity)
        ll::dense_tiled_warp_rm<OUT, H2>(W3, b3, h2, (float *)tb->y, ll::ActIdentity {}, WARPS_PER_BLOCK);

        __syncthreads();
        __threadfence_system();  // make y visible to host before seq_out
        tb->seq_out = last;
    }
}

int main()
{
    ll::check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags(cudaDeviceMapHost)");

    ll::print_device();

    using namespace ll::cfg;
    constexpr int WARPS = ll::cfg::WARPS_PER_BLOCK;
    constexpr int THREADS = WARPS * 32;

    TickBuffer *h_tb = nullptr;
    ll::check_cuda(cudaHostAlloc((void **)&h_tb, sizeof(TickBuffer), cudaHostAllocMapped), "cudaHostAlloc");
    std::memset((void *)h_tb, 0, sizeof(TickBuffer));

    TickBuffer *d_tb = nullptr;
    ll::check_cuda(cudaHostGetDevicePointer((void **)&d_tb, (void *)h_tb, 0), "cudaHostGetDevicePointer");

    float *dW1, *db1, *dW2, *db2, *dW3, *db3;
    ll::check_cuda(cudaMalloc(&dW1, sizeof(float) * H1 * IN), "cudaMalloc W1");
    ll::check_cuda(cudaMalloc(&db1, sizeof(float) * H1), "cudaMalloc b1");
    ll::check_cuda(cudaMalloc(&dW2, sizeof(float) * H2 * H1), "cudaMalloc W2");
    ll::check_cuda(cudaMalloc(&db2, sizeof(float) * H2), "cudaMalloc b2");
    ll::check_cuda(cudaMalloc(&dW3, sizeof(float) * OUT * H2), "cudaMalloc W3");
    ll::check_cuda(cudaMalloc(&db3, sizeof(float) * OUT), "cudaMalloc b3");

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    auto fill = [&](std::vector<float> &v) {
        for (auto &x : v)
            x = dist(rng);
    };

    std::vector<float> hW1(H1 * IN), hb1(H1), hW2(H2 * H1), hb2(H2), hW3(OUT * H2), hb3(OUT);
    fill(hW1);
    fill(hb1);
    fill(hW2);
    fill(hb2);
    fill(hW3);
    fill(hb3);

    ll::check_cuda(cudaMemcpy(dW1, hW1.data(), hW1.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D W1");
    ll::check_cuda(cudaMemcpy(db1, hb1.data(), hb1.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D b1");
    ll::check_cuda(cudaMemcpy(dW2, hW2.data(), hW2.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D W2");
    ll::check_cuda(cudaMemcpy(db2, hb2.data(), hb2.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D b2");
    ll::check_cuda(cudaMemcpy(dW3, hW3.data(), hW3.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D W3");
    ll::check_cuda(cudaMemcpy(db3, hb3.data(), hb3.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D b3");

    cudaStream_t stream;
    ll::check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    persistent_mlp_kernel<<<1, THREADS, 0, stream>>>(d_tb, dW1, db1, dW2, db2, dW3, db3);
    ll::check_cuda(cudaGetLastError(), "kernel launch");

    // Warmup a bit
    for (int w = 1; w <= 1000; ++w) {
        for (int i = 0; i < IN; ++i)
            h_tb->x[i] = dist(rng);
        h_tb->seq_in = (uint32_t)w;
        while (h_tb->seq_out != (uint32_t)w)
            cpu_pause();
        (void)h_tb->y[0];
    }

    // Benchmark
    const int iters = 20000;
    std::vector<double> times;
    times.reserve(iters);

    for (int t = 1; t <= iters; ++t) {
        for (int i = 0; i < IN; ++i)
            h_tb->x[i] = dist(rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t seq = (uint32_t)(1000 + t);
        h_tb->seq_in = seq;
        while (h_tb->seq_out != seq)
            cpu_pause();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times.push_back(us);
    }

    auto st = ll::compute_stats(times);
    ll::print_stats("persistent_mlp_end2end (zero-copy mapped, 1 kernel forever)", st, "us");

    // Stop kernel
    h_tb->stop = 1;
    ll::check_cuda(cudaStreamSynchronize(stream), "sync stop");

    cudaStreamDestroy(stream);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dW3);
    cudaFree(db3);
    cudaFreeHost(h_tb);
    return 0;
}