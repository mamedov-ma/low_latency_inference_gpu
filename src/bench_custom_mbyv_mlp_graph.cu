#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_mbyv.cuh"

using namespace netcfg;

// ---- Compile-time config helpers for MByV ----
// MByV stages a per-thread slice of weights into a user-provided Storage.
// For these layer sizes, the slice does NOT fit in registers/shared memory,
// so we use global scratch (GLOB_N) to preserve correctness.

template <size_t ROWS, size_t COLS, size_t BLOCKS, size_t WARPS>
struct MbyvCfg {
    // Storage type only used for accessing MByV's compile-time constants.
    using DummyStor = GlobMemStor<1, float>;
    using M = MByV<DummyStor, ROWS, COLS, BLOCKS, WARPS>;

    static constexpr size_t padded_cols = ll::PaddedCols<COLS, WARPS>::padded_cols;

    static constexpr size_t big_weights = M::BIG_ROW_ITER_CNT_PER_THREAD * M::COLUMNS_CNT_PER_WARP;
    static constexpr size_t tail_big =
        M::TAIL_ROWS_CNT_PER_BLOCK * (M::FEATS_CNT_PER_THREAD > 0 ? (M::FEATS_CNT_PER_THREAD - 1) : 0);
    static constexpr size_t tail_small = (M::HAS_SMALL_BLOCK_ITERATIONS && M::HAS_SMALL_WARP_ITERATIONS)
                                             ? (M::task_split.cbi_cnt * M::task_split.columns_per_thread)
                                             : 0;

    static constexpr size_t weights_per_thread = big_weights + tail_big + tail_small;
    static_assert(weights_per_thread > 0, "weights_per_thread must be > 0");
};

static std::vector<float> pad_row_major(const std::vector<float> &W_rm, int m, int k, int padded_k)
{
    std::vector<float> Wp((size_t)m * (size_t)padded_k, 0.0f);
    for (int i = 0; i < m; i++) {
        const float *src = &W_rm[(size_t)i * (size_t)k];
        float *dst = &Wp[(size_t)i * (size_t)padded_k];
        for (int j = 0; j < k; j++)
            dst[j] = src[j];
    }
    return Wp;
}

static void host_init(std::vector<float> &v, float scale, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto &x : v)
        x = nd(rng);
}

static std::vector<float> cpu_gemv_bias_relu(const std::vector<float> &W_rm,  // row-major [m x k]
                                             const std::vector<float> &b,     // [m]
                                             const std::vector<float> &x,     // [k]
                                             int m, int k, bool relu)
{
    std::vector<float> y(m, 0.0f);
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float *row = &W_rm[i * k];
        for (int j = 0; j < k; j++)
            acc += row[j] * x[j];
        acc += b[i];
        if (relu)
            acc = acc > 0.0f ? acc : 0.0f;
        y[i] = acc;
    }
    return y;
}

int main()
{
    print_device_info();

    // --- Host weights/biases/input (row-major weights) ---
    std::vector<float> h_x(IN_DIM);
    std::vector<float> h_W1(H1_DIM * IN_DIM), h_b1(H1_DIM);
    std::vector<float> h_W2(H2_DIM * H1_DIM), h_b2(H2_DIM);
    std::vector<float> h_W3(OUT_DIM * H2_DIM), h_b3(OUT_DIM);

    host_init(h_x, 1.0f, 1);
    host_init(h_W1, 0.02f, 2);
    host_init(h_b1, 0.01f, 3);
    host_init(h_W2, 0.02f, 4);
    host_init(h_b2, 0.01f, 5);
    host_init(h_W3, 0.02f, 6);
    host_init(h_b3, 0.01f, 7);

    // --- CPU reference (single run) ---
    auto y1 = cpu_gemv_bias_relu(h_W1, h_b1, h_x, (int)H1_DIM, (int)IN_DIM, true);
    auto y2 = cpu_gemv_bias_relu(h_W2, h_b2, y1, (int)H2_DIM, (int)H1_DIM, true);
    auto y3 = cpu_gemv_bias_relu(h_W3, h_b3, y2, (int)OUT_DIM, (int)H2_DIM, false);

    // ---- MByV kernel configs per layer ----
    // NOTE: This MByV-based path is included mainly for completeness.
    // For these dimensions it is expected to be slower than the `custom_warp` kernel,
    // because it stages per-thread weights into global scratch on every inference.
    using C1 = MbyvCfg<H1_DIM, IN_DIM, 4, 8>;   // layer1: 256x512
    using C2 = MbyvCfg<H2_DIM, H1_DIM, 2, 4>;   // layer2: 64x256
    using C3 = MbyvCfg<OUT_DIM, H2_DIM, 1, 1>;  // layer3: 1x64

    // Pad weights for MByV's expected layout: row-major with padded columns.
    auto h_W1p = pad_row_major(h_W1, (int)H1_DIM, (int)IN_DIM, (int)C1::padded_cols);
    auto h_W2p = pad_row_major(h_W2, (int)H2_DIM, (int)H1_DIM, (int)C2::padded_cols);
    auto h_W3p = pad_row_major(h_W3, (int)OUT_DIM, (int)H2_DIM, (int)C3::padded_cols);

    // --- Device buffers ---
    float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr, *d_y3 = nullptr;
    float *d_W1p = nullptr, *d_W2p = nullptr, *d_W3p = nullptr;
    float *d_b1 = nullptr, *d_b2 = nullptr, *d_b3 = nullptr;
    float *d_scratch1 = nullptr, *d_scratch2 = nullptr, *d_scratch3 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y3, OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1p, (size_t)H1_DIM * C1::padded_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2p, (size_t)H2_DIM * C2::padded_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3p, (size_t)OUT_DIM * C3::padded_cols * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_b1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, (size_t)OUT_DIM * sizeof(float)));

    constexpr size_t scratch1_elems = 4 * 8 * 32 * C1::weights_per_thread;
    constexpr size_t scratch2_elems = 2 * 4 * 32 * C2::weights_per_thread;
    constexpr size_t scratch3_elems = 1 * 1 * 32 * C3::weights_per_thread;
    CUDA_CHECK(cudaMalloc(&d_scratch1, scratch1_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch2, scratch2_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch3, scratch3_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1p, h_W1p.data(), h_W1p.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2p, h_W2p.data(), h_W2p.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3p, h_W3p.data(), h_W3p.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream {};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Inference: 3x fused GEMV+bias+act (MByV-based)
    auto infer_once = [&](cudaStream_t s) {
        {
            dim3 block(32, 8, 1);
            dim3 grid(4, 1, 1);
            ll::dense_mbyv_kernel<H1_DIM, IN_DIM, 4, 8, 0, 0, C1::weights_per_thread, ll::ActRelu>
                <<<grid, block, 0, s>>>(d_W1p, d_b1, d_x, d_y1, d_scratch1, ll::ActRelu {});
        }
        {
            dim3 block(32, 4, 1);
            dim3 grid(2, 1, 1);
            ll::dense_mbyv_kernel<H2_DIM, H1_DIM, 2, 4, 0, 0, C2::weights_per_thread, ll::ActRelu>
                <<<grid, block, 0, s>>>(d_W2p, d_b2, d_y1, d_y2, d_scratch2, ll::ActRelu {});
        }
        {
            dim3 block(32, 1, 1);
            dim3 grid(1, 1, 1);
            ll::dense_mbyv_kernel<OUT_DIM, H2_DIM, 1, 1, 0, 0, C3::weights_per_thread, ll::ActIdentity>
                <<<grid, block, 0, s>>>(d_W3p, d_b3, d_y2, d_y3, d_scratch3, ll::ActIdentity {});
        }
    };

    // --- Correctness (direct) ---
    infer_once(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_out(OUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y3, OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    for (int i = 0; i < (int)OUT_DIM; i++) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_out[i] - (double)y3[i]));
    }
    std::cout << "Correctness: max_abs_err=" << max_abs_err << " (float32 CPU ref)\n";

    // --- Build CUDA Graph ---
    cudaGraph_t graph {};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    infer_once(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graph_exec {};
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // --- Bench direct vs graph ---
    const int warmup = 2000;
    const int iters = 20000;

    // Direct
    {
        auto times = bench_stream_us([&](cudaStream_t s) { infer_once(s); }, stream, warmup, iters, 1);
        auto p = compute_percentiles(times);
        std::cout << "custom_mbyv_mlp_fp32 (direct, 3x fused GEMV+bias+act)";
        print_stats("", p);
    }

    // Graph
    {
        auto times = bench_stream_us([&](cudaStream_t s) { CUDA_CHECK(cudaGraphLaunch(graph_exec, s)); }, stream,
                                     warmup, iters, 1);
        auto p = compute_percentiles(times);
        std::cout << "custom_mbyv_mlp_fp32 (CUDA Graph, single launch)";
        print_stats("", p);
    }

    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_y3));
    CUDA_CHECK(cudaFree(d_W1p));
    CUDA_CHECK(cudaFree(d_W2p));
    CUDA_CHECK(cudaFree(d_W3p));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_b3));

    CUDA_CHECK(cudaFree(d_scratch1));
    CUDA_CHECK(cudaFree(d_scratch2));
    CUDA_CHECK(cudaFree(d_scratch3));

    return 0;
}
