
#pragma once
#include <cstddef>

// fixed model shape for batch=1 (compile-time friendly).
namespace netcfg {
static constexpr std::size_t IN_DIM = 512;
static constexpr std::size_t H1_DIM = 256;
static constexpr std::size_t H2_DIM = 64;
static constexpr std::size_t OUT_DIM = 1;

// For latency studies we keep batch=1 always.
static constexpr std::size_t BATCH = 1;
}  // namespace netcfg

// Network config for persistent-kernel demo.
// Chosen to keep intermediates small while still realistic.
namespace ll::cfg {
constexpr int IN = 512;
constexpr int H1 = 256;
constexpr int H2 = 64;
constexpr int OUT = 1;

// Persistent kernel block configuration
constexpr int WARPS_PER_BLOCK = 8;  // 8 warps = 256 threads
}  // namespace ll::cfg
