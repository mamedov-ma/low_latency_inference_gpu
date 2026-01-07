// src/bench_persistent_cublas_mlp.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "bench_utils.h"
#include "net_config.h"

using namespace netcfg;

#define CUBLAS_CHECK(call)                                                                                 \
    do {                                                                                                   \
        cublasStatus_t _s = (call);                                                                        \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::cerr << "cuBLAS error " << int(_s) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

// -------------------------
// Host-side helpers
// -------------------------
static inline void cpu_pause()
{
#if defined(__x86_64__) || defined(_M_X64)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

static inline void host_sfence()
{
#if defined(__x86_64__) || defined(_M_X64)
    __asm__ __volatile__("sfence" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
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

// -------------------------
// TickBuffer (host pinned + mapped)
// -------------------------
struct alignas(64) TickBuffer {
    volatile uint32_t seq_in;
    volatile uint32_t seq_out;
    volatile uint32_t stop;
    uint32_t pad;
    float x[IN_DIM];
    float y[OUT_DIM];
};

// -------------------------
// Device helpers for sysmem mapped loads/stores
// -------------------------
__device__ __forceinline__ uint32_t ld_cv_u32(const volatile uint32_t *p)
{
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float ld_cv_f32(const float *p)
{
    float v;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void st_wt_u32(volatile uint32_t *p, uint32_t v)
{
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
}

// -------------------------
// Small kernels: stage x from mapped host to device, write y back to mapped host
// -------------------------
__global__ void stage_x_from_mapped(float *__restrict__ d_x, const TickBuffer *__restrict__ tb)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < (int)IN_DIM; i += stride)
        d_x[i] = ld_cv_f32(&tb->x[i]);
}

__global__ void write_y_to_mapped(TickBuffer *__restrict__ tb, const float *__restrict__ d_y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < (int)OUT_DIM; i += stride)
        tb->y[i] = d_y[i];
}

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

// Publish completion to host (seq_out) after y is written.
// Must be enqueued AFTER write_y_to_mapped on the same stream.
__global__ void publish_seq_out(TickBuffer *__restrict__ tb, uint32_t s)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();   // make y[] visible to host before seq_out
        st_wt_u32(&tb->seq_out, s);
        __threadfence_system();   // ensure seq_out itself visible promptly
    }
}

int main()
{
    // MUST be before context init to enable mapped host alloc
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    print_device_info();

    // Host weights/biases/input (row-major)
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

    // CPU reference
    auto y1_ref = cpu_gemv_bias_relu(h_W1, h_b1, h_x, (int)H1_DIM, (int)IN_DIM, true);
    auto y2_ref = cpu_gemv_bias_relu(h_W2, h_b2, y1_ref, (int)H2_DIM, (int)H1_DIM, true);
    auto y3_ref = cpu_gemv_bias_relu(h_W3, h_b3, y2_ref, (int)OUT_DIM, (int)H2_DIM, false);

    // Device buffers
    float *d_x=nullptr, *d_y1=nullptr, *d_y2=nullptr, *d_y3=nullptr;
    float *d_W1=nullptr, *d_b1=nullptr, *d_W2=nullptr, *d_b2=nullptr, *d_W3=nullptr, *d_b3=nullptr;

    CUDA_CHECK(cudaMalloc(&d_x,  IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y3, OUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1, h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, h_b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, h_W3.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, h_b3.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // cuBLAS setup
    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // TickBuffer pinned+mapped (+WC helps CPU->GPU writes)
    TickBuffer *h_tb=nullptr;
    CUDA_CHECK(cudaHostAlloc((void **)&h_tb, sizeof(TickBuffer),
                             cudaHostAllocMapped | cudaHostAllocWriteCombined));
    std::memset((void *)h_tb, 0, sizeof(TickBuffer));

    TickBuffer *d_tb=nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_tb, (void *)h_tb, 0));

    // Initial input
    for (size_t i = 0; i < IN_DIM; i++)
        h_tb->x[i] = h_x[i];

    // Enqueue one inference (uses mapped buffer for x/y + publishes seq_out)
    auto infer_enqueue = [&](uint32_t s) {
        constexpr float alpha = 1.0f;
        constexpr float beta  = 0.0f;

        // Stage x: sysmem(mapped) -> device
        stage_x_from_mapped<<<((int)IN_DIM + 255) / 256, 256, 0, stream>>>(d_x, d_tb);

        // Layer1: y1 = W1*x  (W1 row-major [H1 x IN] => use opT)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)H1_DIM, 1, (int)IN_DIM,
                                 &alpha,
                                 d_W1, (int)IN_DIM,
                                 d_x,  (int)IN_DIM,
                                 &beta,
                                 d_y1, (int)H1_DIM));
        bias_relu<<<((int)H1_DIM + 255) / 256, 256, 0, stream>>>(d_y1, d_b1, (int)H1_DIM);

        // Layer2
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)H2_DIM, 1, (int)H1_DIM,
                                 &alpha,
                                 d_W2, (int)H1_DIM,
                                 d_y1, (int)H1_DIM,
                                 &beta,
                                 d_y2, (int)H2_DIM));
        bias_relu<<<((int)H2_DIM + 255) / 256, 256, 0, stream>>>(d_y2, d_b2, (int)H2_DIM);

        // Layer3 (no relu)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)OUT_DIM, 1, (int)H2_DIM,
                                 &alpha,
                                 d_W3, (int)H2_DIM,
                                 d_y2, (int)H2_DIM,
                                 &beta,
                                 d_y3, (int)OUT_DIM));
        bias_only<<<((int)OUT_DIM + 255) / 256, 256, 0, stream>>>(d_y3, d_b3, (int)OUT_DIM);

        // Write y back: device -> sysmem(mapped)
        write_y_to_mapped<<<((int)OUT_DIM + 255) / 256, 256, 0, stream>>>(d_tb, d_y3);

        // Publish completion
        publish_seq_out<<<1, 1, 0, stream>>>(d_tb, s);
    };

    // -------------------------
    // Correctness: 1 tick
    // -------------------------
    uint32_t seq = 1;
    std::atomic_thread_fence(std::memory_order_release);
    h_tb->seq_in = seq;
    host_sfence();

    infer_enqueue(seq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Wait host-visible seq_out (should already be updated after synchronize, but keep symmetry)
    while (h_tb->seq_out != seq) cpu_pause();

    double max_abs_err = 0.0;
    for (int i = 0; i < (int)OUT_DIM; i++)
        max_abs_err = std::max(max_abs_err, std::abs((double)h_tb->y[i] - (double)y3_ref[i]));
    std::cout << "Correctness: max_abs_err=" << max_abs_err << " (float32 CPU ref)\n";

    // -------------------------
    // End-to-end latency benchmark
    // -------------------------
    const int warmup = 2000;
    const int iters  = 20000;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        std::atomic_thread_fence(std::memory_order_release);
        h_tb->seq_in = ++seq;
        host_sfence();

        infer_enqueue(seq);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        while (h_tb->seq_out != seq) cpu_pause();
    }

    std::vector<double> e2e_us;
    e2e_us.reserve(iters);

    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();

        std::atomic_thread_fence(std::memory_order_release);
        h_tb->seq_in = ++seq;
        host_sfence();

        infer_enqueue(seq);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        while (h_tb->seq_out != seq) cpu_pause();

        auto t1 = std::chrono::steady_clock::now();
        e2e_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto st = ll::compute_stats(e2e_us);
    ll::print_stats("persistent_style_cublas_mlp_fp32 (mapped seq_in/seq_out + cublas)", st, "us");

    // Stop (symmetry)
    h_tb->stop = 1;
    std::atomic_thread_fence(std::memory_order_release);
    h_tb->seq_in = ++seq;
    host_sfence();

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_y3));

    CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_W3)); CUDA_CHECK(cudaFree(d_b3));

    CUDA_CHECK(cudaFreeHost(h_tb));
    return 0;
}
