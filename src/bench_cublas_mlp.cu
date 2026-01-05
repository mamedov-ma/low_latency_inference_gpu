
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "bench_utils.h"
#include "net_config.h"

#define CUBLAS_CHECK(call)                                                                                 \
    do {                                                                                                   \
        cublasStatus_t _s = (call);                                                                        \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::cerr << "cuBLAS error " << int(_s) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

__global__ void bias_relu(float *__restrict__ y, const float *__restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = y[i] + b[i];
        y[i] = (v > 0.0f) ? v : 0.0f;
    }
}

__global__ void bias_only(float *__restrict__ y, const float *__restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += b[i];
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
    using namespace netcfg;
    print_device_info();

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

    CUDA_CHECK(cudaMalloc(&d_W1, h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, h_b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, h_W3.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, h_b3.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream {};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // --- cuBLAS setup ---
    cublasHandle_t handle {};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    auto infer_once = [&](cudaStream_t s) {
        (void)s;
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Layer1: y1 = W1*x  (W1 row-major [H1 x IN])
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)H1_DIM, 1, (int)IN_DIM, &alpha, d_W1,
                                 (int)IN_DIM, d_x, (int)IN_DIM, &beta, d_y1, (int)H1_DIM));
        int t1 = ((int)H1_DIM + 255) / 256;
        bias_relu<<<t1, 256, 0, stream>>>(d_y1, d_b1, (int)H1_DIM);

        // Layer2
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)H2_DIM, 1, (int)H1_DIM, &alpha, d_W2,
                                 (int)H1_DIM, d_y1, (int)H1_DIM, &beta, d_y2, (int)H2_DIM));
        int t2 = ((int)H2_DIM + 255) / 256;
        bias_relu<<<t2, 256, 0, stream>>>(d_y2, d_b2, (int)H2_DIM);

        // Layer3 (no relu for output)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)OUT_DIM, 1, (int)H2_DIM, &alpha, d_W3,
                                 (int)H2_DIM, d_y2, (int)H2_DIM, &beta, d_y3, (int)OUT_DIM));
        int t3 = ((int)OUT_DIM + 255) / 256;
        bias_only<<<t3, 256, 0, stream>>>(d_y3, d_b3, (int)OUT_DIM);
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

    // warmup
    for (int i = 0; i < warmup; i++)
        infer_once(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto times = bench_stream_us(
        [&](cudaStream_t s) {
            (void)s;
            infer_once(stream);
        },
        stream, warmup, iters,
        /*inner_reps=*/1);

    auto p = compute_percentiles(times);
    std::cout << "cublas_mlp_fp32 (3xGEMM + bias/act kernels) [us] "
              << "p50=" << p.p50 << " p90=" << p.p90 << " p99=" << p.p99 << " min=" << p.min << " max=" << p.max
              << " mean=" << p.mean << " sd=" << p.stdev << "\n";

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
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
