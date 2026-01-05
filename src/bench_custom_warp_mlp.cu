#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "bench_utils.h"
#include "net_config.h"
#include "ll/dense_warp.cuh"

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

    // --- Device buffers ---
    float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr, *d_y3 = nullptr;
    float *d_W1 = nullptr, *d_b1 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr, *d_W3 = nullptr, *d_b3 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y3, OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)H1_DIM * IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)H2_DIM * H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)OUT_DIM * H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, (size_t)OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream {};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Configs for T4-ish GPUs. We'll tune later if needed.
    constexpr int BW1 = 8;  // warps per block for layer1 (256 outputs)
    constexpr int BW2 = 4;  // warps per block for layer2 (64 outputs)
    constexpr int BW3 = 1;  // 1 output

    auto infer_once = [&](cudaStream_t s) {
        // Layer1: y1 = ReLU(W1*x + b1)
        {
            dim3 block(BW1 * 32);
            dim3 grid((H1_DIM + BW1 - 1) / BW1);
            size_t shmem = IN_DIM * sizeof(float);
            ll::dense_warp_rm_kernel<(int)H1_DIM, (int)IN_DIM, BW1>
                <<<grid, block, shmem, s>>>(d_W1, d_b1, d_x, d_y1, ll::ActRelu {});
        }
        // Layer2: y2 = ReLU(W2*y1 + b2)
        {
            dim3 block(BW2 * 32);
            dim3 grid((H2_DIM + BW2 - 1) / BW2);
            size_t shmem = H1_DIM * sizeof(float);
            ll::dense_warp_rm_kernel<(int)H2_DIM, (int)H1_DIM, BW2>
                <<<grid, block, shmem, s>>>(d_W2, d_b2, d_y1, d_y2, ll::ActRelu {});
        }
        // Layer3: y3 = W3*y2 + b3
        {
            dim3 block(BW3 * 32);
            dim3 grid((OUT_DIM + BW3 - 1) / BW3);
            size_t shmem = H2_DIM * sizeof(float);
            ll::dense_warp_rm_kernel<(int)OUT_DIM, (int)H2_DIM, BW3>
                <<<grid, block, shmem, s>>>(d_W3, d_b3, d_y2, d_y3, ll::ActIdentity {});
        }
    };

    // --- Correctness check ---
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
    std::cout << "custom_warp_mlp_fp32 (3x warp-row GEMV+bias+act) [us] "
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
    return 0;
}
