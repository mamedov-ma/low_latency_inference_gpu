// src/bench_persistent_mbyv_mlp.cu
#include <cuda_runtime.h>

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
#include "ll/dense_mbyv.cuh"

using namespace netcfg;

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
    // Flush write-combining buffers / order WC writes (important for CPU->GPU doorbell).
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
// Device helpers for mapped host polling
// IMPORTANT: use .cv loads for sysmem to avoid L2 "sticking".
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

// device mem reads/writes (use atomics to avoid compiler caching)
__device__ __forceinline__ uint32_t ld_u32(const uint32_t *p)
{
    return (uint32_t)atomicAdd((unsigned int *)p, 0u);
}
__device__ __forceinline__ void st_u32(uint32_t *p, uint32_t v)
{
    atomicExch((unsigned int *)p, (unsigned int)v);
}

// host-visible store (works for sysmem when paired with __threadfence_system)
__device__ __forceinline__ void st_wt_u32(volatile uint32_t *p, uint32_t v)
{
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
}

__device__ __forceinline__ void relax()
{
#if __CUDA_ARCH__ >= 700
    __nanosleep(200);
#endif
}

// -------------------------
// Persistent kernel
// -------------------------
__global__ void persistent_mbyv_mlp_kernel(
    TickBuffer *tb,
    const float *__restrict__ W1, const float *__restrict__ b1,
    const float *__restrict__ W2, const float *__restrict__ b2,
    const float *__restrict__ W3, const float *__restrict__ b3,
    float *__restrict__ d_x,
    float *__restrict__ y1,
    float *__restrict__ y2,
    float *__restrict__ scratch1,
    // device-side sync
    uint32_t *__restrict__ work_seq,      // published by block0 AFTER staging input
    uint32_t *__restrict__ l1_done_seq,   // published by block0 after L1 all blocks done
    uint32_t *__restrict__ l2_done_seq,   // published by block0 after L2 blocks done
    uint32_t *__restrict__ tick_done_seq, // published by block0 after L3 done
    uint32_t *__restrict__ l1_cnt,
    uint32_t *__restrict__ l2_cnt)
{
    constexpr uint32_t STOP_SEQ = 0xFFFFFFFFu;

    // L1: 512 -> 256
    constexpr int L1_BLOCKS = 4;
    constexpr int L1_WARPS  = 8;
    constexpr int L1_REG_N  = 32;
    constexpr int L1_SHM_N  = 44;
    constexpr int L1_GLOB_N = 128 - L1_REG_N - L1_SHM_N; // 52

    // L2: 256 -> 64
    constexpr int L2_BLOCKS = 2;
    constexpr int L2_WARPS  = 8;
    constexpr int L2_REG_N  = 32;
    constexpr int L2_SHM_N  = 0;
    constexpr int L2_GLOB_N = 0;

    // L3: 64 -> 1
    constexpr int L3_BLOCKS = 1;
    constexpr int L3_WARPS  = 8;
    constexpr int L3_REG_N  = 32;
    constexpr int L3_SHM_N  = 0;
    constexpr int L3_GLOB_N = 0;

    const int bid  = (int)blockIdx.x;   // 0..3
    const int lane = (int)threadIdx.x;  // 0..31
    const int warp = (int)threadIdx.y;  // 0..7

    __shared__ float l1_shmem[L1_SHM_N * L1_WARPS * 32];

    // block0-only shared tick latch
    __shared__ volatile uint32_t sh_stage_s;

    if (bid == 0) {
        if (lane == 0 && warp == 0) sh_stage_s = 0;
        __syncthreads();
    }

    // --------- Build MByV objects (weights loaded once) ---------
    float *l1_sh_ptr = &l1_shmem[(warp * 32 + lane) * L1_SHM_N];
    float *l1_g_ptr  = nullptr;
    if constexpr (L1_GLOB_N > 0) {
        const size_t tid_linear =
            ((size_t)bid * (size_t)L1_WARPS + (size_t)warp) * 32ull + (size_t)lane;
        l1_g_ptr = &scratch1[tid_linear * (size_t)L1_GLOB_N];
    }

    auto l1_ws = ::make_weights_storage<L1_REG_N, L1_SHM_N, L1_GLOB_N>(l1_sh_ptr, l1_g_ptr);
    using L1WS = decltype(l1_ws);
    using L1M  = ::MByV<L1WS, H1_DIM, IN_DIM, (size_t)L1_BLOCKS, (size_t)L1_WARPS>;
    L1M l1(l1_ws);

    auto l2_ws = ::make_weights_storage<L2_REG_N, L2_SHM_N, L2_GLOB_N>(nullptr, nullptr);
    using L2WS = decltype(l2_ws);
    using L2M  = ::MByV<L2WS, H2_DIM, H1_DIM, (size_t)L2_BLOCKS, (size_t)L2_WARPS>;
    L2M l2(l2_ws);

    auto l3_ws = ::make_weights_storage<L3_REG_N, L3_SHM_N, L3_GLOB_N>(nullptr, nullptr);
    using L3WS = decltype(l3_ws);
    using L3M  = ::MByV<L3WS, OUT_DIM, H2_DIM, (size_t)L3_BLOCKS, (size_t)L3_WARPS>;
    L3M l3(l3_ws);

    // Load weights ONCE
    l1.load_kernel(W1, b1);
    if (bid < L2_BLOCKS) l2.load_kernel(W2, b2);
    // if (bid == 0)        l3.load_kernel(W3, b3);

    uint32_t last_tick = 0;

    while (true) {
        // -----------------------------
        // block0: poll seq_in, stage x, then publish work_seq
        // -----------------------------
        if (bid == 0) {
            // leader polls and latches tick into shared
            if (lane == 0 && warp == 0) {
                for (;;) {
                    if (ld_cv_u32(&tb->stop)) {
                        sh_stage_s = STOP_SEQ;
                        break;
                    }
                    uint32_t s_in = ld_cv_u32(&tb->seq_in);
                    if (s_in != 0 && s_in != last_tick) {
                        // reset only counters per tick
                        st_u32(l1_cnt, 0);
                        st_u32(l2_cnt, 0);
                        __threadfence();
                        sh_stage_s = s_in;
                        break;
                    }
                    relax();
                }
            }

            // all block0 threads wait for latch
            uint32_t s_lat;
            while ((s_lat = sh_stage_s) == 0u) relax();

            // stop path: publish STOP and exit
            if (s_lat == STOP_SEQ) {
                if (lane == 0 && warp == 0) {
                    st_u32(work_seq, STOP_SEQ);
                    st_u32(l1_done_seq, STOP_SEQ);
                    st_u32(l2_done_seq, STOP_SEQ);
                    st_u32(tick_done_seq, STOP_SEQ);
                    __threadfence_system();
                }
                return;
            }

            // stage input using all block0 threads (use .cv reads from sysmem)
            int tid    = warp * 32 + lane;
            int stride = (int)(blockDim.x * blockDim.y); // 256
            for (int i = tid; i < (int)IN_DIM; i += stride)
                d_x[i] = ld_cv_f32(&tb->x[i]);

            __syncthreads();
            __threadfence();

            // publish work_seq after staging; clear latch for next tick
            if (lane == 0 && warp == 0) {
                st_u32(work_seq, s_lat);
                sh_stage_s = 0u;
            }
            __syncthreads();
        }

        // -----------------------------
        // all blocks: wait for work_seq to advance
        // -----------------------------
        uint32_t s;
        for (;;) {
            s = ld_u32(work_seq);
            if (s == STOP_SEQ) return;
            if (s != 0 && s != last_tick) break;
            relax();
        }

        // -------------------------
        // L1: all 4 blocks run
        // -------------------------
        {
            float feats[L1M::FEATS_CNT_PER_THREAD];
            constexpr size_t ACC_N =
                L1M::BIG_ROW_ITER_CNT_PER_THREAD + L1M::P2_TASKS_CNT + L1M::task_split.cbi_cnt;
            float acc[ACC_N];

            l1.load_feats(d_x, feats);
            l1.calc(acc, feats);
            l1.save_result_act(acc, y1, ll::ActRelu{});
            __threadfence();

            if (lane == 0 && warp == 0)
                atomicAdd((unsigned int *)l1_cnt, 1u);

            if (bid == 0 && lane == 0 && warp == 0) {
                while (ld_u32(l1_cnt) != (uint32_t)L1_BLOCKS) relax();
                st_u32(l1_done_seq, s);
                __threadfence();
            }
        }

        // -------------------------
        // L2: only blocks 0..1 run
        // -------------------------
        if (bid < L2_BLOCKS) {
            while (ld_u32(l1_done_seq) < s) {
                if (ld_u32(work_seq) == STOP_SEQ) return;
                relax();
            }

            float feats[L2M::FEATS_CNT_PER_THREAD];
            constexpr size_t ACC_N =
                L2M::BIG_ROW_ITER_CNT_PER_THREAD + L2M::P2_TASKS_CNT + L2M::task_split.cbi_cnt;
            float acc[ACC_N];

            l2.load_feats(y1, feats);
            l2.calc(acc, feats);
            l2.save_result_act(acc, y2, ll::ActRelu{});
            __threadfence();

            if (lane == 0 && warp == 0)
                atomicAdd((unsigned int *)l2_cnt, 1u);

            if (bid == 0 && lane == 0 && warp == 0) {
                while (ld_u32(l2_cnt) != (uint32_t)L2_BLOCKS) relax();
                st_u32(l2_done_seq, s);
                __threadfence();
            }
        }

        // -------------------------
        // L3: only block0 runs
        // -------------------------
        if (bid == 0) {
            while (ld_u32(l2_done_seq) < s) {
                if (ld_u32(work_seq) == STOP_SEQ) return;
                relax();
            }

            float feats[L3M::FEATS_CNT_PER_THREAD];
            constexpr size_t ACC_N =
                L3M::BIG_ROW_ITER_CNT_PER_THREAD + L3M::P2_TASKS_CNT + L3M::task_split.cbi_cnt;
            float acc[ACC_N];

            // l3.load_feats(y2, feats);
            // l3.calc(acc, feats);
            l3.save_result_act(acc, (float *)tb->y, ll::ActIdentity{});

            __threadfence_system();         // y visible to host
            if (lane == 0 && warp == 0) {
                st_wt_u32(&tb->seq_out, s); // host-visible completion
                st_u32(tick_done_seq, s);   // device-visible completion barrier
                __threadfence_system();
            }
        }

        // end-of-tick barrier: monotonic
        while (ld_u32(tick_done_seq) < s) {
            if (ld_u32(work_seq) == STOP_SEQ) return;
            relax();
        }

        last_tick = s;
    }
}

// -------------------------
// Main
// -------------------------
int main()
{
    // MUST be before context init to enable mapped host alloc
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    print_device_info();

    // Host weights/biases/input (row-major weights, unpadded)
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

    // Tunings
    constexpr size_t L1_BLOCKS = 4, L1_WARPS = 8;
    constexpr size_t L2_BLOCKS = 2, L2_WARPS = 8;
    constexpr size_t L3_BLOCKS = 1, L3_WARPS = 8;

    // Pack weights into padded layout expected by MByV
    auto h_W1pad = pack_weights_padded_rm<(int)H1_DIM, (int)IN_DIM, (int)L1_WARPS>(h_W1);
    auto h_W2pad = pack_weights_padded_rm<(int)H2_DIM, (int)H1_DIM, (int)L2_WARPS>(h_W2);
    auto h_W3pad = pack_weights_padded_rm<(int)OUT_DIM, (int)H2_DIM, (int)L3_WARPS>(h_W3);

    constexpr size_t W1_PAD_COLS = ll::PaddedCols<IN_DIM, L1_WARPS>::padded_cols;
    constexpr size_t W2_PAD_COLS = ll::PaddedCols<H1_DIM, L2_WARPS>::padded_cols;
    constexpr size_t W3_PAD_COLS = ll::PaddedCols<H2_DIM, L3_WARPS>::padded_cols;

    // Device buffers
    float *d_x=nullptr, *d_y1=nullptr, *d_y2=nullptr;
    float *d_W1=nullptr, *d_b1=nullptr, *d_W2=nullptr, *d_b2=nullptr, *d_W3=nullptr, *d_b3=nullptr;
    float *d_scratch1=nullptr;

    CUDA_CHECK(cudaMalloc(&d_x,  IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, H2_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)H1_DIM * W1_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, (size_t)H1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)H2_DIM * W2_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, (size_t)H2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)OUT_DIM * W3_PAD_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, (size_t)OUT_DIM * sizeof(float)));

    // L1 scratch
    constexpr size_t L1_REG_N = 32, L1_SHMEM_N = 44, L1_GLOB_N = 128 - L1_REG_N - L1_SHMEM_N; // 52
    if constexpr (L1_GLOB_N > 0) {
        CUDA_CHECK(cudaMalloc(&d_scratch1,
                              (size_t)L1_BLOCKS * (size_t)L1_WARPS * 32ull * (size_t)L1_GLOB_N * sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpy(d_W1, h_W1pad.data(), h_W1pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(),   h_b1.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2pad.data(), h_W2pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(),   h_b2.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W3, h_W3pad.data(), h_W3pad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(),   h_b3.size()   * sizeof(float), cudaMemcpyHostToDevice));

    // TickBuffer pinned+mapped
    TickBuffer *h_tb=nullptr;
    CUDA_CHECK(cudaHostAlloc((void **)&h_tb, sizeof(TickBuffer),
                             cudaHostAllocMapped | cudaHostAllocWriteCombined));
    std::memset((void *)h_tb, 0, sizeof(TickBuffer));

    TickBuffer *d_tb=nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_tb, (void *)h_tb, 0));

    // Device sync vars
    uint32_t *d_work_seq=nullptr, *d_l1_done=nullptr, *d_l2_done=nullptr, *d_tick_done=nullptr, *d_l1_cnt=nullptr, *d_l2_cnt=nullptr;
    CUDA_CHECK(cudaMalloc(&d_work_seq,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_l1_done,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_l2_done,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_tick_done, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_l1_cnt,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_l2_cnt,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_work_seq,  0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_l1_done,   0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_done,   0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_tick_done, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_l1_cnt,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_cnt,    0, sizeof(uint32_t)));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Launch persistent kernel: grid=4 blocks, block=(32,8)
    dim3 block(32, (unsigned)L1_WARPS, 1);
    persistent_mbyv_mlp_kernel<<<(int)L1_BLOCKS, block, 0, stream>>>(
        d_tb,
        d_W1, d_b1,
        d_W2, d_b2,
        d_W3, d_b3,
        d_x, d_y1, d_y2, d_scratch1,
        d_work_seq, d_l1_done, d_l2_done, d_tick_done, d_l1_cnt, d_l2_cnt);
    CUDA_CHECK(cudaGetLastError());

    volatile uint32_t *seq_out_p = &h_tb->seq_out;

    // -------------------------
    // Correctness: 1 tick
    // -------------------------
    for (size_t i = 0; i < IN_DIM; i++)
        h_tb->x[i] = h_x[i];

    uint32_t seq = 1;
    std::atomic_thread_fence(std::memory_order_release);
    h_tb->seq_in = seq;
    host_sfence();

    while (*seq_out_p != seq) {
        cpu_pause();
    }

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
        // std::cout << "warmup iter: " << i << "\n";
        std::atomic_thread_fence(std::memory_order_release);
        h_tb->seq_in = ++seq;
        host_sfence();
        while (*seq_out_p != seq)
            cpu_pause();
    }

    std::vector<double> e2e_us;
    e2e_us.reserve(iters);

    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();

        std::atomic_thread_fence(std::memory_order_release);
        h_tb->seq_in = ++seq;
        host_sfence();

        while (*seq_out_p != seq)
            cpu_pause();

        auto t1 = std::chrono::steady_clock::now();
        e2e_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto st = ll::compute_stats(e2e_us);
    ll::print_stats("persistent_mbyv_mlp_fp32 (mapped seq_in/seq_out, 1 persistent kernel)", st, "us");

    // Stop
    h_tb->stop = 1;
    std::atomic_thread_fence(std::memory_order_release);
    h_tb->seq_in = ++seq;
    host_sfence();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_y2));

    CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_W3)); CUDA_CHECK(cudaFree(d_b3));

    if (d_scratch1) CUDA_CHECK(cudaFree(d_scratch1));

    CUDA_CHECK(cudaFree(d_work_seq));
    CUDA_CHECK(cudaFree(d_l1_done));
    CUDA_CHECK(cudaFree(d_l2_done));
    CUDA_CHECK(cudaFree(d_tick_done));
    CUDA_CHECK(cudaFree(d_l1_cnt));
    CUDA_CHECK(cudaFree(d_l2_cnt));

    CUDA_CHECK(cudaFreeHost(h_tb));
    return 0;
}
