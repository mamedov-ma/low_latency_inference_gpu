#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_mbyv.cuh"

using namespace netcfg;

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

// Pack row-major weights [M x K] into row-major padded weights [M x Kpad]
// Kpad = WARPS * ceil_div(K, WARPS)

template <int M, int K, int WARPS>
static std::vector<float> pack_weights_padded_rm(const std::vector<float> &W_rm)
{
    constexpr int cols_per_warp = (K + WARPS - 1) / WARPS;
    constexpr int Kpad = WARPS * cols_per_warp;
    std::vector<float> W_pad((size_t)M * (size_t)Kpad, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            W_pad[(size_t)i * (size_t)Kpad + (size_t)j] = W_rm[(size_t)i * (size_t)K + (size_t)j];
        }
    }
    return W_pad;
}

int main()
{
    print_device_info();

    // --- Host weights/biases/input (row-major weights, unpadded) ---
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

    // --- Custom kernel configs (tuned for small batch=1 on T4; tweak later) ---
    // Layer1: 512 -> 256
    constexpr size_t L1_BLOCKS = 4;
    constexpr size_t L1_WARPS = 8;
    // Per-thread weights: rows_per_thread * cols_per_warp = ceil(256/(4*32)) * ceil(512/8) = 2 * 64 = 128
    constexpr size_t L1_REG_N = 16;
    constexpr size_t L1_SHMEM_N = 16;
    constexpr size_t L1_GLOB_N = 128 - L1_REG_N - L1_SHMEM_N;

    // Layer2: 256 -> 64
    constexpr size_t L2_BLOCKS = 2;
    constexpr size_t L2_WARPS = 8;
    // ceil(64/(2*32)) * ceil(256/8) = 1 * 32 = 32
    constexpr size_t L2_REG_N = 16;
    constexpr size_t L2_SHMEM_N = 8;
    constexpr size_t L2_GLOB_N = 32 - L2_REG_N - L2_SHMEM_N;

    // Layer3: 64 -> 1
    constexpr size_t L3_BLOCKS = 1;
    constexpr size_t L3_WARPS = 2;
    // ceil(1/(1*32)) * ceil(64/2) = 1 * 32 = 32
    constexpr size_t L3_REG_N = 16;
    constexpr size_t L3_SHMEM_N = 0;
    constexpr size_t L3_GLOB_N = 32 - L3_REG_N - L3_SHMEM_N;

    // --- Pack weights into padded layout expected by MByV kernels ---
    auto h_W1pad = pack_weights_padded_rm<(int)H1_DIM, (int)IN_DIM, (int)L1_WARPS>(h_W1);
    auto h_W2pad = pack_weights_padded_rm<(int)H2_DIM, (int)H1_DIM, (int)L2_WARPS>(h_W2);
    auto h_W3pad = pack_weights_padded_rm<(int)OUT_DIM, (int)H2_DIM, (int)L3_WARPS>(h_W3);

    constexpr size_t W1_PAD_COLS = ll::PaddedCols<IN_DIM, L1_WARPS>::padded_cols;
    constexpr size_t W2_PAD_COLS = ll::PaddedCols<H1_DIM, L2_WARPS>::padded_cols;
    constexpr size_t W3_PAD_COLS = ll::PaddedCols<H2_DIM, L3_WARPS>::padded_cols;

    // --- Device buffers ---
    float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr, *d_y3 = nullptr;
    float *d_W1 = nullptr, *d_b1 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr, *d_W3 = nullptr, *d_b3 = nullptr;
    float *d_scratch1 = nullptr, *d_scratch2 = nullptr, *d_scratch3 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y3, OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)H1_DIM * W1_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)H2_DIM * W2_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)OUT_DIM * W3_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, (size_t)OUT_DIM * sizeof(float)));

    if constexpr (L1_GLOB_N > 0)
        CUDA_CHECK(cudaMalloc(&d_scratch1, (size_t)L1_BLOCKS * L1_WARPS * 32 * L1_GLOB_N * sizeof(float)));
    if constexpr (L2_GLOB_N > 0)
        CUDA_CHECK(cudaMalloc(&d_scratch2, (size_t)L2_BLOCKS * L2_WARPS * 32 * L2_GLOB_N * sizeof(float)));
    if constexpr (L3_GLOB_N > 0)
        CUDA_CHECK(cudaMalloc(&d_scratch3, (size_t)L3_BLOCKS * L3_WARPS * 32 * L3_GLOB_N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1pad.data(), h_W1pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2pad.data(), h_W2pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3pad.data(), h_W3pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream {};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto infer_once = [&](cudaStream_t s) {
        // Layer1
        dim3 b1(32, (unsigned)L1_WARPS, 1);
        ll::dense_mbyv_kernel<H1_DIM, IN_DIM, L1_BLOCKS, L1_WARPS, L1_REG_N, L1_SHMEM_N, L1_GLOB_N>
            <<<L1_BLOCKS, b1, 0, s>>>(d_W1, d_b1, d_x, d_y1, d_scratch1, ll::ActRelu {});

        // Layer2
        dim3 b2(32, (unsigned)L2_WARPS, 1);
        ll::dense_mbyv_kernel<H2_DIM, H1_DIM, L2_BLOCKS, L2_WARPS, L2_REG_N, L2_SHMEM_N, L2_GLOB_N>
            <<<L2_BLOCKS, b2, 0, s>>>(d_W2, d_b2, d_y1, d_y2, d_scratch2, ll::ActRelu {});

        // Layer3 (no relu)
        dim3 b3(32, (unsigned)L3_WARPS, 1);
        ll::dense_mbyv_kernel<OUT_DIM, H2_DIM, L3_BLOCKS, L3_WARPS, L3_REG_N, L3_SHMEM_N, L3_GLOB_N>
            <<<L3_BLOCKS, b3, 0, s>>>(d_W3, d_b3, d_y2, d_y3, d_scratch3, ll::ActIdentity {});
    };

    // --- Correctness check (single run) ---
    infer_once(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_out(OUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y3, OUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    for (int i = 0; i < (int)OUT_DIM; i++) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_out[i] - (double)y3[i]));
    }
    std::cout << "Correctness: max_abs_err=" << max_abs_err << " (float32 CPU ref)\n";

    // --- Benchmark ---
    const int warmup = 2000;
    const int iters = 20000;

    for (int i = 0; i < warmup; i++)
        infer_once(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto times = bench_stream_us([&](cudaStream_t s) { infer_once(s); }, stream, warmup, iters,
                                 /*inner_reps=*/1);

    auto p = compute_percentiles(times);
    std::cout << "custom_mbyv_mlp_fp32 (3x fused GEMV+bias+act) [us] "
              << "p50=" << p.p50 << " p90=" << p.p90 << " p99=" << p.p99 << " min=" << p.min << " max=" << p.max
              << " mean=" << p.mean << " sd=" << p.stdev << "\n";

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_y3));

    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_W3));
    CUDA_CHECK(cudaFree(d_b3));

    if (d_scratch1)
        CUDA_CHECK(cudaFree(d_scratch1));
    if (d_scratch2)
        CUDA_CHECK(cudaFree(d_scratch2));
    if (d_scratch3)
        CUDA_CHECK(cudaFree(d_scratch3));

    return 0;
}
