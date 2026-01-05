#include <type_traits>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <utility>

// decomposition of a natural number into powers of 2
template <size_t N>
consteval std::array<size_t, std::popcount(N)> gen_p2_tasks()
{
    // static_assert(N > 0, "N must be positive");
    if constexpr (N == 0) {
        return {};
    }

    std::array<size_t, std::popcount(N)> out {};
    size_t idx = std::popcount(N);

    for (size_t b = 0; b < std::numeric_limits<size_t>::digits; ++b) {
        if ((N >> b) & 1u) {
            out[--idx] = size_t {1} << b;
        }
    }
    return out;
}

// unrolls a loop with a constexpr bound at compile time;
// this is useful because the index also becomes constexpr
template <size_t N, class F>
__device__ __forceinline__ void static_for(F &&f)
{
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f(std::integral_constant<size_t, Is> {}), ...);
    }(std::make_index_sequence<N> {});
}

consteval size_t ceil_div(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

consteval size_t roundup(size_t x, size_t m)
{
    return m == 0 ? x : ((x + m - 1) / m) * m;
}

// round up to an even number
consteval size_t roundup_even(size_t x)
{
    return x + (x & 1u);
}

// maximum power of two that divides x (2^k | x)
consteval size_t max_pow2_divisor(size_t x)
{
    if (x == 0) {
        return 0;
    }
    size_t p = 1;
    while ((x % (p << 1u)) == 0) {
        p <<= 1u;
    }
    return p;
}

namespace detail {

template <size_t R, size_t C>
consteval bool can_finish()
{
    constexpr size_t kWarpSize = 32;

    if constexpr (R == 0 || C == 0)
        return true;
    if constexpr (C == 1)
        return true;
    if constexpr (R == 1)
        return true;
    if constexpr (C == kWarpSize)
        return true;

    size_t rows_ans = roundup_even(R);
    size_t cols_ans = roundup_even(C);

    auto prod = [](size_t r, size_t c) { return r * c; };

    size_t weights = prod(rows_ans, cols_ans);
    if (weights % kWarpSize == 0ull)
        return true;

    size_t target = roundup((size_t)(weights % std::numeric_limits<size_t>::max()), kWarpSize);

    for (size_t i = 0; i < 1'000'000; ++i) {
        size_t rows_inc = (size_t)ceil_div((size_t)target, cols_ans);
        size_t cols_inc = (size_t)ceil_div((size_t)target, rows_ans);

        size_t area0 = prod(rows_inc, cols_ans);
        size_t area1 = prod(rows_ans, cols_inc);
        size_t area2 = prod(rows_inc, cols_inc);

        size_t optimal = 0;
        size_t best = area0;
        if (area1 < best) {
            best = area1;
            optimal = 1;
        }
        if (area2 < best) {
            best = area2;
            optimal = 2;
        }

        size_t rows_tmp = rows_ans, cols_tmp = cols_ans;
        if (optimal == 0)
            rows_tmp = rows_inc;
        else if (optimal == 1)
            cols_tmp = cols_inc;
        else {
            rows_tmp = rows_inc;
            cols_tmp = cols_inc;
        }

        if ((rows_tmp % 2u) == 0u && (cols_tmp % 2u) == 0u && (prod(rows_tmp, cols_tmp) % kWarpSize) == 0ull)
            return true;

        target += kWarpSize;
    }
    return false;  // “not found” within the limit
}

}  // namespace detail

template <size_t rows_cnt, size_t cols_cnt>
consteval std::pair<size_t, size_t> get_task_dim_cbi_cwi()
{
    static_assert(detail::can_finish<rows_cnt, cols_cnt>(),
                  "get_task_dim_cbi_cwi(): internal search failed — bug or bad params");

    constexpr size_t kWarpSize = 32;

    if constexpr (rows_cnt == 0 || cols_cnt == 0)
        return {0, 0};
    if constexpr (cols_cnt == 1)
        return {kWarpSize, 1};
    if constexpr (rows_cnt == 1)
        return {1, kWarpSize};
    if constexpr (cols_cnt == kWarpSize)
        return {rows_cnt, cols_cnt};

    size_t rows_ans = roundup_even(rows_cnt);
    size_t cols_ans = roundup_even(cols_cnt);

    auto prod = [](size_t r, size_t c) { return r * c; };

    size_t weights = prod(rows_ans, cols_ans);
    if (weights % kWarpSize == 0ull)
        return {rows_ans, cols_ans};

    size_t target = roundup((size_t)(weights % std::numeric_limits<size_t>::max()), kWarpSize);
    for (size_t i = 0; i < 1'000'000; ++i) {
        size_t rows_inc = (size_t)ceil_div((size_t)target, cols_ans);
        size_t cols_inc = (size_t)ceil_div((size_t)target, rows_ans);

        size_t area0 = prod(rows_inc, cols_ans);
        size_t area1 = prod(rows_ans, cols_inc);
        size_t area2 = prod(rows_inc, cols_inc);

        size_t optimal = 0;
        size_t best = area0;
        if (area1 < best) {
            best = area1;
            optimal = 1;
        }
        if (area2 < best) {
            best = area2;
            optimal = 2;
        }

        size_t rows_tmp = rows_ans, cols_tmp = cols_ans;
        if (optimal == 0)
            rows_tmp = rows_inc;
        else if (optimal == 1)
            cols_tmp = cols_inc;
        else {
            rows_tmp = rows_inc;
            cols_tmp = cols_inc;
        }

        if ((rows_tmp % 2u) == 0u && (cols_tmp % 2u) == 0u && (prod(rows_tmp, cols_tmp) % kWarpSize) == 0ull)
            return {rows_tmp, cols_tmp};

        target += kWarpSize;
    }

#if __cpp_lib_unreachable >= 202202L
    std::unreachable();  // C++23
#else
    __builtin_unreachable();  // GCC/Clang
#endif
}

struct Split {
    size_t cbi_cnt;
    size_t rows_in_one_block;
    size_t columns_per_thread;
};

template <size_t TAIL_ROWS_CNT_PER_BLOCK, size_t WEIGHTS_FOR_SMALL_WARP_ITER>
consteval Split get_task_split_cbi_cwi()
{
    constexpr size_t kWarpSize = 32;
    constexpr auto task_dim = get_task_dim_cbi_cwi<TAIL_ROWS_CNT_PER_BLOCK, WEIGHTS_FOR_SMALL_WARP_ITER>();
    constexpr size_t rows_cnt = task_dim.first;
    constexpr size_t cols_cnt = task_dim.second;

    if constexpr (rows_cnt == 0 || cols_cnt == 0) {
        return {0, 0, 0};
    }
    static_assert(rows_cnt % 2u == 0u || rows_cnt == 1u, "invalid input rows cnt: must be 1 or multiple of 2");

    constexpr size_t rp2 = (rows_cnt == 1) ? 1 : max_pow2_divisor(rows_cnt);
    constexpr size_t cbi_cnt = (rp2 == 0) ? 0 : (rows_cnt / rp2);
    constexpr size_t columns_per_thread = (rp2 * cols_cnt) / kWarpSize;
    constexpr size_t lhs = kWarpSize * cbi_cnt * columns_per_thread;
    constexpr size_t rhs = rows_cnt * cols_cnt;

    static_assert(lhs == rhs, "invalid split: {32 x cbi_cnt x columns_per_thread} must equalto {rows_cnt x cols_cnt}");
    return {cbi_cnt, rp2, columns_per_thread};
}
