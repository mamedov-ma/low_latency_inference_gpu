#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_warp.cuh"  // ActRelu/ActIdentity
#include "ll/dense_mbyv_packed.cuh"
#include "ll/pack_weights.cuh"

using namespace netcfg;

static void host_init(std::vector<float> &v, float scale, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto &x : v)
        x = nd(rng);
}

static void cpu_mlp_ref(const std::vector<float> &W1, const std::vector<float> &b1, const std::vector<float> &W2,
                        const std::vector<float> &b2, const std::vector<float> &W3, const std::vector<float> &b3,
                        const std::vector<float> &x, std::vector<float> &y)
{
    std::vector<float> h1(H1_DIM), h2(H2_DIM);
    // layer1
    for (int m = 0; m < H1_DIM; ++m) {
        float acc = b1[m];
        const float *row = &W1[m * IN_DIM];
        for (int k = 0; k < IN_DIM; ++k)
            acc += row[k] * x[k];
        h1[m] = acc > 0.f ? acc : 0.f;
    }
    // layer2
    for (int m = 0; m < H2_DIM; ++m) {
        float acc = b2[m];
        const float *row = &W2[m * H1_DIM];
        for (int k = 0; k < H1_DIM; ++k)
            acc += row[k] * h1[k];
        h2[m] = acc > 0.f ? acc : 0.f;
    }
    // layer3
    for (int m = 0; m < OUT_DIM; ++m) {
        float acc = b3[m];
        const float *row = &W3[m * H2_DIM];
        for (int k = 0; k < H2_DIM; ++k)
            acc += row[k] * h2[k];
        y[m] = acc;
    }
}

static float max_abs_err(const std::vector<float> &a, const std::vector<float> &b)
{
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

int main()
{
    print_device_info();

    // Host weights (row-major) + bias
    std::vector<float> W1(H1_DIM * IN_DIM), b1(H1_DIM);
    std::vector<float> W2(H2_DIM * H1_DIM), b2(H2_DIM);
    std::vector<float> W3(OUT_DIM * H2_DIM), b3(OUT_DIM);
    std::vector<float> x(IN_DIM), y_ref(OUT_DIM), y_out(OUT_DIM);

    host_init(W1, 0.02f, 1);
    host_init(b1, 0.01f, 2);
    host_init(W2, 0.02f, 3);
    host_init(b2, 0.01f, 4);
    host_init(W3, 0.02f, 5);
    host_init(b3, 0.01f, 6);
    host_init(x, 1.00f, 7);

    cpu_mlp_ref(W1, b1, W2, b2, W3, b3, x, y_ref);

    // Pack weights to [K, Mpad] for coalesced thread-per-row loads
    constexpr int H1_PAD = (H1_DIM + 31) / 32 * 32;
    constexpr int H2_PAD = (H2_DIM + 31) / 32 * 32;
    constexpr int OUT_PAD = (OUT_DIM + 31) / 32 * 32;

    std::vector<float> W1p(IN_DIM * H1_PAD), b1p(H1_PAD);
    std::vector<float> W2p(H1_DIM * H2_PAD), b2p(H2_PAD);
    std::vector<float> W3p(H2_DIM * OUT_PAD), b3p(OUT_PAD);

    ll::pack_rm_to_kmajor_padded(W1.data(), H1_DIM, IN_DIM, W1p.data(), H1_PAD);
    ll::pack_bias_padded(b1.data(), H1_DIM, b1p.data(), H1_PAD);
    ll::pack_rm_to_kmajor_padded(W2.data(), H2_DIM, H1_DIM, W2p.data(), H2_PAD);
    ll::pack_bias_padded(b2.data(), H2_DIM, b2p.data(), H2_PAD);
    ll::pack_rm_to_kmajor_padded(W3.data(), OUT_DIM, H2_DIM, W3p.data(), OUT_PAD);
    ll::pack_bias_padded(b3.data(), OUT_DIM, b3p.data(), OUT_PAD);

    // Device buffers
    float *dW1 = 0, *db1 = 0, *dW2 = 0, *db2 = 0, *dW3 = 0, *db3 = 0, *dx = 0, *dy1 = 0, *dy2 = 0, *dy = 0;
    CUDA_CHECK(cudaMalloc(&dW1, sizeof(float) * W1p.size()));
    CUDA_CHECK(cudaMalloc(&db1, sizeof(float) * b1p.size()));
    CUDA_CHECK(cudaMalloc(&dW2, sizeof(float) * W2p.size()));
    CUDA_CHECK(cudaMalloc(&db2, sizeof(float) * b2p.size()));
    CUDA_CHECK(cudaMalloc(&dW3, sizeof(float) * W3p.size()));
    CUDA_CHECK(cudaMalloc(&db3, sizeof(float) * b3p.size()));
    CUDA_CHECK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CUDA_CHECK(cudaMalloc(&dy1, sizeof(float) * H1_DIM));
    CUDA_CHECK(cudaMalloc(&dy2, sizeof(float) * H2_DIM));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(float) * OUT_DIM));

    CUDA_CHECK(cudaMemcpy(dW1, W1p.data(), sizeof(float) * W1p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, b1p.data(), sizeof(float) * b1p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2p.data(), sizeof(float) * W2p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, b2p.data(), sizeof(float) * b2p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW3, W3p.data(), sizeof(float) * W3p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db3, b3p.data(), sizeof(float) * b3p.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- Correctness run (direct) ----
    ll::launch_dense_mbyv_packed<H1_DIM, IN_DIM>(stream, dW1, db1, dx, dy1, ll::ActRelu {});
    ll::launch_dense_mbyv_packed<H2_DIM, H1_DIM>(stream, dW2, db2, dy1, dy2, ll::ActRelu {});
    ll::launch_dense_mbyv_packed<OUT_DIM, H2_DIM>(stream, dW3, db3, dy2, dy, ll::ActIdentity {});
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(y_out.data(), dy, sizeof(float) * OUT_DIM, cudaMemcpyDeviceToHost));
    std::cout << "Correctness: max_abs_err=" << max_abs_err(y_ref, y_out) << " (float32 CPU ref)\n";

    // ---- Bench: direct launches ----
    auto direct_fn = [&](cudaStream_t s) {
        ll::launch_dense_mbyv_packed<H1_DIM, IN_DIM>(s, dW1, db1, dx, dy1, ll::ActRelu {});
        ll::launch_dense_mbyv_packed<H2_DIM, H1_DIM>(s, dW2, db2, dy1, dy2, ll::ActRelu {});
        ll::launch_dense_mbyv_packed<OUT_DIM, H2_DIM>(s, dW3, db3, dy2, dy, ll::ActIdentity {});
    };
    auto r_direct = bench_stream_us(direct_fn, stream, /*warmup=*/2000, /*iters=*/20000);
    print_stats("custom_mbyv_packed_fp32 (direct, thread-per-row + packed W)", compute_percentiles(r_direct));

    // ---- CUDA Graph capture (3 kernels) ----
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    direct_fn(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // graph correctness
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(y_out.data(), dy, sizeof(float) * OUT_DIM, cudaMemcpyDeviceToHost));
    std::cout << "Graph correctness: max_abs_err=" << max_abs_err(y_ref, y_out) << " (float32 CPU ref)\n";

    auto graph_fn = [&](cudaStream_t s) { CUDA_CHECK(cudaGraphLaunch(graph_exec, s)); };
    auto r_graph = bench_stream_us(graph_fn, stream, /*warmup=*/2000, /*iters=*/20000);
    print_stats("custom_mbyv_packed_fp32 (CUDA Graph, 1 launch)", compute_percentiles(r_graph));

    // cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dW1));
    CUDA_CHECK(cudaFree(db1));
    CUDA_CHECK(cudaFree(dW2));
    CUDA_CHECK(cudaFree(db2));
    CUDA_CHECK(cudaFree(dW3));
    CUDA_CHECK(cudaFree(db3));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy1));
    CUDA_CHECK(cudaFree(dy2));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
