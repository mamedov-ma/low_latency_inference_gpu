
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        cudaError_t _e = (call);                                                                             \
        if (_e != cudaSuccess) {                                                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " (" << static_cast<int>(_e) << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;                                           \
            std::exit(1);                                                                                    \
        }                                                                                                    \
    } while (0)

struct Percentiles {
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stdev = 0.0;
};

inline Percentiles compute_percentiles(std::vector<double> us)
{
    assert(!us.empty());
    std::sort(us.begin(), us.end());

    auto pick = [&](double p) -> double {
        // nearest-rank method
        double idx = p * (us.size() - 1);
        std::size_t i = static_cast<std::size_t>(idx + 0.5);
        if (i >= us.size())
            i = us.size() - 1;
        return us[i];
    };

    Percentiles out {};
    out.p50 = pick(0.50);
    out.p90 = pick(0.90);
    out.p99 = pick(0.99);
    out.min = us.front();
    out.max = us.back();
    out.mean = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());

    double var = 0.0;
    for (double x : us)
        var += (x - out.mean) * (x - out.mean);
    var /= static_cast<double>(us.size());
    out.stdev = std::sqrt(var);
    return out;
}

inline void print_stats(const std::string &name, const Percentiles &s)
{
    std::cout << name << " [us] "
              << "p50=" << s.p50 << " p90=" << s.p90 << " p99=" << s.p99 << " min=" << s.min << " max=" << s.max
              << " mean=" << s.mean << " sd=" << s.stdev << std::endl;
}

inline void print_device_info()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p {};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::cout << "CUDA device: " << p.name << "\n"
              << "  SMs: " << p.multiProcessorCount << "  CC: " << p.major << "." << p.minor << "\n"
              << "  clockRate(kHz): " << p.clockRate << "\n"
              << "  memClockRate(kHz): " << p.memoryClockRate << "\n"
              << "  memoryBusWidth(bits): " << p.memoryBusWidth << "\n"
              << "  globalMem(GB): " << (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << "\n"
              << std::endl;
}

// Measure time of a function that enqueues work on the given stream.
// We use cudaEvent on the same stream to capture kernel-only or stream-scoped sequences.
// Measure time of a function that enqueues work on the given stream.
// We use cudaEvent on the same stream to capture kernel-only or stream-scoped sequences.
//
// inner_reps > 1 is useful to amortize cudaEvent overhead for tiny kernels;
// returned time is normalized per 1 enqueue call.
template <class EnqueueFn>
inline std::vector<double> bench_stream_us(EnqueueFn &&enqueue, cudaStream_t stream, int warmup_iters, int iters,
                                           int inner_reps = 1)
{
    if (inner_reps < 1)
        inner_reps = 1;
    std::vector<double> us;
    us.reserve(iters);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        for (int r = 0; r < inner_reps; ++r)
            enqueue(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int r = 0; r < inner_reps; ++r)
            enqueue(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double per_call_us = (static_cast<double>(ms) * 1000.0) / static_cast<double>(inner_reps);
        us.push_back(per_call_us);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return us;
}

namespace ll {

inline void check_cuda(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Stats {
    double p50 = 0, p90 = 0, p99 = 0, min = 0, max = 0, mean = 0, sd = 0;
};

inline Stats compute_stats(std::vector<double> xs)
{
    Stats s {};
    if (xs.empty())
        return s;
    std::sort(xs.begin(), xs.end());
    auto pct = [&](double p) -> double {
        if (xs.size() == 1)
            return xs[0];
        double idx = p * (xs.size() - 1);
        size_t i = (size_t)std::floor(idx);
        size_t j = std::min(i + 1, xs.size() - 1);
        double t = idx - i;
        return xs[i] * (1.0 - t) + xs[j] * t;
    };
    s.p50 = pct(0.50);
    s.p90 = pct(0.90);
    s.p99 = pct(0.99);
    s.min = xs.front();
    s.max = xs.back();
    s.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
    double var = 0.0;
    for (double v : xs)
        var += (v - s.mean) * (v - s.mean);
    var /= xs.size();
    s.sd = std::sqrt(var);
    return s;
}

inline void print_stats(const char *name, const Stats &s, const char *unit = "us")
{
    std::printf("%s [%s] p50=%.3f p90=%.3f p99=%.3f min=%.3f max=%.3f mean=%.3f sd=%.3f\n", name, unit, s.p50, s.p90,
                s.p99, s.min, s.max, s.mean, s.sd);
}

inline void print_device()
{
    int dev = 0;
    check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp p {};
    check_cuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    std::printf("CUDA device: %s\n", p.name);
    std::printf("  SMs: %d  CC: %d.%d\n", p.multiProcessorCount, p.major, p.minor);
    std::printf("  clockRate(kHz): %d\n", p.clockRate);
    std::printf("  memClockRate(kHz): %d\n", p.memoryClockRate);
    std::printf("  memoryBusWidth(bits): %d\n", p.memoryBusWidth);
    std::printf("  globalMem(GB): %.4f\n\n", double(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
}

}  // namespace ll
