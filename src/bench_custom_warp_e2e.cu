#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_warp.cuh"

// End-to-end benchmark for batch=1:
// H2D memcpy + 3x fused warp-row dense kernels + D2H memcpy.
// Compares: pageable host mem vs pinned, and direct launches vs CUDA Graph (pinned only).

using namespace netcfg;

static void fill_input(float *x, int n, uint32_t seed)
{
    // cheap deterministic "tick" data; avoids denormals
    for (int i = 0; i < n; ++i) {
        uint32_t v = seed * 1664525u + 1013904223u + (uint32_t)i;
        float f = (float)((v & 0xFFFFu) - 32768) / 32768.0f;
        x[i] = f;
    }
}

// NOTE: dense_warp_rm_kernel is fully compile-time for M and K.
// This is intentional for low-latency: fixed sizes let the compiler unroll
// and avoid runtime branches.
template <int M, int K, int WARPS_PER_BLOCK, class Act>
inline void launch_dense(cudaStream_t s, const float *W_rm, const float *b, const float *x, float *y, Act act)
{
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t smem = (size_t)K * sizeof(float);  // stage x
    ll::dense_warp_rm_kernel<M, K, WARPS_PER_BLOCK, Act><<<grid, block, smem, s>>>(W_rm, b, x, y, act);
}

struct Buffers {
    // host
    float *h_x = nullptr;
    float *h_y = nullptr;
    bool pinned = false;

    // device
    float *d_x = nullptr;
    float *d_y1 = nullptr;
    float *d_y2 = nullptr;
    float *d_y3 = nullptr;

    // weights (device)
    float *d_W1 = nullptr;
    float *d_b1 = nullptr;
    float *d_W2 = nullptr;
    float *d_b2 = nullptr;
    float *d_W3 = nullptr;
    float *d_b3 = nullptr;
};

static void alloc_host(Buffers &bufs, bool pinned)
{
    bufs.pinned = pinned;
    if (pinned) {
        CUDA_CHECK(cudaHostAlloc((void **)&bufs.h_x, (size_t)IN_DIM * sizeof(float), cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc((void **)&bufs.h_y, (size_t)OUT_DIM * sizeof(float), cudaHostAllocPortable));
    } else {
        bufs.h_x = (float *)std::malloc((size_t)IN_DIM * sizeof(float));
        bufs.h_y = (float *)std::malloc((size_t)OUT_DIM * sizeof(float));
    }
    if (!bufs.h_x || !bufs.h_y) {
        std::cerr << "Host allocation failed\n";
        std::exit(1);
    }
    std::memset(bufs.h_y, 0, (size_t)OUT_DIM * sizeof(float));
}

static void free_host(Buffers &bufs)
{
    if (bufs.pinned) {
        if (bufs.h_x)
            CUDA_CHECK(cudaFreeHost(bufs.h_x));
        if (bufs.h_y)
            CUDA_CHECK(cudaFreeHost(bufs.h_y));
    } else {
        std::free(bufs.h_x);
        std::free(bufs.h_y);
    }
    bufs.h_x = bufs.h_y = nullptr;
}

static void alloc_device(Buffers &b)
{
    CUDA_CHECK(cudaMalloc(&b.d_x, (size_t)IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_y1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_y2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_y3, (size_t)OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&b.d_W1, (size_t)H1_DIM * IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_b1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_W2, (size_t)H2_DIM * H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_b2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_W3, (size_t)OUT_DIM * H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_b3, (size_t)OUT_DIM * sizeof(float)));
}

static void free_device(Buffers &b)
{
    auto f = [&](void *p) {
        if (p)
            CUDA_CHECK(cudaFree(p));
    };
    f(b.d_x);
    f(b.d_y1);
    f(b.d_y2);
    f(b.d_y3);
    f(b.d_W1);
    f(b.d_b1);
    f(b.d_W2);
    f(b.d_b2);
    f(b.d_W3);
    f(b.d_b3);
    std::memset(&b, 0, sizeof(Buffers));
}

static void init_weights(Buffers &b, cudaStream_t stream)
{
    // Create deterministic weights on host (pageable is fine; one-time).
    std::vector<float> W1((size_t)H1_DIM * IN_DIM);
    std::vector<float> b1((size_t)H1_DIM);
    std::vector<float> W2((size_t)H2_DIM * H1_DIM);
    std::vector<float> b2((size_t)H2_DIM);
    std::vector<float> W3((size_t)OUT_DIM * H2_DIM);
    std::vector<float> b3((size_t)OUT_DIM);

    auto fillW = [&](std::vector<float> &W, uint32_t seed) {
        for (size_t i = 0; i < W.size(); ++i) {
            uint32_t v = seed * 1664525u + 1013904223u + (uint32_t)i;
            float f = (float)((v & 0xFFFFu) - 32768) / 32768.0f;
            W[i] = f * 0.05f;  // keep outputs reasonable
        }
    };
    auto fillB = [&](std::vector<float> &B, uint32_t seed) {
        for (size_t i = 0; i < B.size(); ++i) {
            uint32_t v = seed * 22695477u + 1u + (uint32_t)i;
            float f = (float)((v & 0xFFFFu) - 32768) / 32768.0f;
            B[i] = f * 0.01f;
        }
    };

    fillW(W1, 1);
    fillB(b1, 11);
    fillW(W2, 2);
    fillB(b2, 22);
    fillW(W3, 3);
    fillB(b3, 33);

    CUDA_CHECK(cudaMemcpyAsync(b.d_W1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.d_W2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.d_W3, W3.data(), W3.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.d_b3, b3.data(), b3.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void run_mlp_direct(Buffers &b, cudaStream_t s)
{
    CUDA_CHECK(cudaMemcpyAsync(b.d_x, b.h_x, (size_t)IN_DIM * sizeof(float), cudaMemcpyHostToDevice, s));

    // Tuned warps-per-block per layer for T4-ish GPUs (CC 7.5).
    // Layer1: many output rows -> more warps per block.
    // Layer2: fewer rows.
    // Layer3: out_dim=1.
    constexpr int BW1 = 8;
    constexpr int BW2 = 4;
    constexpr int BW3 = 1;

    launch_dense<(int)H1_DIM, (int)IN_DIM, BW1>(s, b.d_W1, b.d_b1, b.d_x, b.d_y1, ll::ActRelu {});
    launch_dense<(int)H2_DIM, (int)H1_DIM, BW2>(s, b.d_W2, b.d_b2, b.d_y1, b.d_y2, ll::ActRelu {});
    launch_dense<(int)OUT_DIM, (int)H2_DIM, BW3>(s, b.d_W3, b.d_b3, b.d_y2, b.d_y3, ll::ActIdentity {});

    CUDA_CHECK(cudaMemcpyAsync(b.h_y, b.d_y3, (size_t)OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, s));
}

static cudaGraphExec_t build_graph(Buffers &b, cudaStream_t s)
{
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    run_mlp_direct(b, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return exec;
}

int main()
{
    print_device_info();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // We'll run 3 configs:
    // 1) direct + pageable
    // 2) direct + pinned
    // 3) CUDA Graph + pinned
    const int warmup = 2000;
    const int iters = 20000;

    auto bench_cfg = [&](const std::string &name, bool pinned, bool use_graph) {
        Buffers b;
        alloc_host(b, pinned);
        alloc_device(b);
        init_weights(b, stream);

        cudaGraphExec_t exec = nullptr;
        if (use_graph) {
            exec = build_graph(b, stream);
        }

        // one correctness run
        fill_input(b.h_x, IN_DIM, 123);
        if (use_graph) {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        } else {
            run_mlp_direct(b, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint32_t seed = 1;
        auto us = bench_stream_us(
            [&](cudaStream_t s) {
                fill_input(b.h_x, IN_DIM, seed++);
                if (use_graph) {
                    CUDA_CHECK(cudaGraphLaunch(exec, s));
                } else {
                    run_mlp_direct(b, s);
                }
            },
            stream, warmup, iters, 1);

        auto stats = compute_percentiles(std::move(us));
        print_stats(name, stats);

        if (exec)
            CUDA_CHECK(cudaGraphExecDestroy(exec));
        free_device(b);
        free_host(b);
    };

    bench_cfg("custom_warp_end2end (direct + pageable H2D/D2H)", false, false);
    bench_cfg("custom_warp_end2end (direct + pinned H2D/D2H)", true, false);
    bench_cfg("custom_warp_end2end (CUDA Graph + pinned, 1x launch)", true, true);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}