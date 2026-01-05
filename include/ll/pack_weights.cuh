#pragma once
#include <cstddef>
#include <cstring>
#include <vector>

namespace ll {

// Round up x to next multiple of m (m must be power of two or any positive)
static inline size_t round_up(size_t x, size_t m)
{
    return ((x + m - 1) / m) * m;
}

// Pack row-major weights W_rm[M,K] into column-major-like layout W_cm[K, Mpad]:
//   W_cm[k*Mpad + m] = (m < M) ? W_rm[m*K + k] : 0
inline void pack_rm_to_kmajor_padded(const float *W_rm, size_t M, size_t K, float *W_cm, size_t Mpad)
{
    // fill zeros
    std::memset(W_cm, 0, sizeof(float) * K * Mpad);
    for (size_t m = 0; m < M; ++m) {
        const float *row = W_rm + m * K;
        for (size_t k = 0; k < K; ++k) {
            W_cm[k * Mpad + m] = row[k];
        }
    }
}

inline void pack_bias_padded(const float *b, size_t M, float *b_pad, size_t Mpad)
{
    std::memset(b_pad, 0, sizeof(float) * Mpad);
    std::memcpy(b_pad, b, sizeof(float) * M);
}

}  // namespace ll
