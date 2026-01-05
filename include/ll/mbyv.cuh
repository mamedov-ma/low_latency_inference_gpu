#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.cuh"

template <class Storage, size_t ROWS_CNT, size_t COLUMNS_CNT, size_t BLOCKS_CNT, size_t WARPS_CNT>
class MByV {
public:
    // Warp size
    static constexpr size_t THREADS_CNT_PER_WARP = 32;
    // How many columns one warp processes
    static constexpr size_t COLUMNS_CNT_PER_WARP = ceil_div(COLUMNS_CNT, WARPS_CNT);
    // The full “big” part in columns, aligned to the warp size.
    static constexpr size_t COLUMNS_CNT_PER_WARP_BIG_ITER =
        COLUMNS_CNT_PER_WARP - COLUMNS_CNT_PER_WARP % THREADS_CNT_PER_WARP;
    // How many “feature segments” (of 32) a thread (lane) loads within a warp.
    static constexpr size_t FEATS_CNT_PER_THREAD = ceil_div(COLUMNS_CNT_PER_WARP, THREADS_CNT_PER_WARP);
    // How many “big” row iterations per thread
    static constexpr size_t BIG_ROW_ITER_CNT_PER_THREAD = ROWS_CNT / (BLOCKS_CNT * THREADS_CNT_PER_WARP);
    // How many rows a block processes (including tail).
    static constexpr size_t ROWS_CNT_PER_BLOCK = ceil_div(ROWS_CNT, BLOCKS_CNT);
    // Row remainder in a block not divisible by a warp (row tail).
    static constexpr size_t TAIL_ROWS_CNT_PER_BLOCK = ROWS_CNT_PER_BLOCK % THREADS_CNT_PER_WARP;
    // Number of “p2 tasks” (tail layout by powers of two).
    static constexpr size_t P2_TASKS_CNT = std::popcount(TAIL_ROWS_CNT_PER_BLOCK);
    // How many columns remain for a “small” warp iteration (column tail).
    static constexpr size_t WEIGHTS_FOR_SMALL_WARP_ITER = COLUMNS_CNT_PER_WARP % THREADS_CNT_PER_WARP;
    static constexpr bool HAS_SMALL_WARP_ITERATIONS = WEIGHTS_FOR_SMALL_WARP_ITER > 0;
    static constexpr bool HAS_SMALL_BLOCK_ITERATIONS = TAIL_ROWS_CNT_PER_BLOCK > 0;
    static constexpr bool HAS_PADDED_ROWS = ROWS_CNT % BLOCKS_CNT > 0;
    static constexpr bool HAS_PADDED_COLUMNS = COLUMNS_CNT % WARPS_CNT > 0;
    static constexpr auto p2_tasks = gen_p2_tasks<TAIL_ROWS_CNT_PER_BLOCK>();
    static constexpr auto task_split = get_task_split_cbi_cwi<TAIL_ROWS_CNT_PER_BLOCK, WEIGHTS_FOR_SMALL_WARP_ITER>();

    __device__ MByV(Storage &ws)
        : ws_(&ws), BLOCK_ID(blockIdx.x), WARP_ID_IN_BLOCK(threadIdx.y), THREAD_ID_IN_WARP(threadIdx.x)
    {
    }

    __forceinline__ __device__ void load_kernel(const float *__restrict__ _weights, const float *__restrict__ _bias)
    {
// “Big” row iteration (without row tail)
#pragma unroll
        for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
            // Global row index for our thread within the block.
            const size_t offset_in_row_axis =
                (BLOCK_ID * THREADS_CNT_PER_WARP + THREAD_ID_IN_WARP) * BIG_ROW_ITER_CNT_PER_THREAD + i;
#pragma unroll
            for (size_t j = 0; j < COLUMNS_CNT_PER_WARP; ++j) {
                // Local index in Storage buffer for the current element.
                const size_t k = i * COLUMNS_CNT_PER_WARP + j;
                // Column offset for the current warp within the block.
                const size_t offset_in_column = WARP_ID_IN_BLOCK * COLUMNS_CNT_PER_WARP + j;
                constexpr size_t PADDED_COLS = WARPS_CNT * COLUMNS_CNT_PER_WARP;
                const size_t index = offset_in_row_axis * PADDED_COLS + offset_in_column;
                if constexpr (HAS_PADDED_COLUMNS) {
                    const bool valid_index = offset_in_column < COLUMNS_CNT;
                    const float w = valid_index ? _weights[index] : 0.0f;
                    ws_->set(k, w);
                } else {
                    const float w = _weights[index];
                    ws_->set(k, w);
                }
            }
            bias_local[i] = _bias[offset_in_row_axis];
        }

        // 2) “Small” row iteration (TAIL_ROWS_CNT_PER_BLOCK tail). Inside: first the “big” column part,
        // then the column tail
        if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
            // Offset inside the local Storage buffer for the tail rows part.
            size_t offset_local = BIG_ROW_ITER_CNT_PER_THREAD * COLUMNS_CNT_PER_WARP;
            size_t lines_processed = 0;
            // 2.1) Split the row tail into p2-tasks (powers of two).
            static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                constexpr size_t I = I_c;
                constexpr size_t n = p2_tasks[I];  // how many rows are processed by the group
                constexpr size_t threads_per_row = THREADS_CNT_PER_WARP / n;
                constexpr size_t columns_per_thread = COLUMNS_CNT_PER_WARP_BIG_ITER / threads_per_row;
                // Global row offset for the tail block.
                const size_t block_offset_global = BLOCKS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD +
                                                   BLOCK_ID * TAIL_ROWS_CNT_PER_BLOCK + lines_processed;
// “Big” column part (warp-aligned)
#pragma unroll
                for (size_t j = 0; j < columns_per_thread; ++j) {
                    const size_t index =
                        ((block_offset_global + THREAD_ID_IN_WARP % n) * WARPS_CNT + WARP_ID_IN_BLOCK) *
                            COLUMNS_CNT_PER_WARP +
                        (THREAD_ID_IN_WARP / n * n) + (j / n * THREADS_CNT_PER_WARP) + j % n;
                    const size_t k = offset_local + j;
                    if constexpr (HAS_PADDED_ROWS) {
                        const bool valid_index = (block_offset_global + THREAD_ID_IN_WARP % n) < ROWS_CNT;
                        const float w = valid_index ? _weights[index] : 0.0f;
                        ws_->set(k, w);
                    } else {
                        const float w = _weights[index];
                        ws_->set(k, w);
                    }
                }
                // Store bias for the corresponding “tail” row
                bias_local[BIG_ROW_ITER_CNT_PER_THREAD + I] = _bias[block_offset_global + THREAD_ID_IN_WARP % n];
                offset_local += columns_per_thread;
                lines_processed += n;
            });

            // 2.2) Column tail inside tail rows (if there is also a column tail)
            if constexpr (HAS_SMALL_BLOCK_ITERATIONS && HAS_SMALL_WARP_ITERATIONS) {
                lines_processed = 0;
#pragma unroll
                for (size_t i = 0; i < task_split.cbi_cnt; ++i) {
                    const size_t block_offset_global = BLOCKS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD +
                                                       BLOCK_ID * TAIL_ROWS_CNT_PER_BLOCK + lines_processed;
#pragma unroll
                    for (size_t j = 0; j < task_split.columns_per_thread; ++j) {
                        const size_t index =
                            ((block_offset_global + THREAD_ID_IN_WARP % task_split.rows_in_one_block) * WARPS_CNT +
                             WARP_ID_IN_BLOCK) *
                                COLUMNS_CNT_PER_WARP +
                            COLUMNS_CNT_PER_WARP_BIG_ITER +
                            (THREAD_ID_IN_WARP / task_split.rows_in_one_block * task_split.columns_per_thread) + j;
                        const bool valid_index =
                            (block_offset_global + THREAD_ID_IN_WARP % task_split.rows_in_one_block < ROWS_CNT) &&
                            (WARP_ID_IN_BLOCK * COLUMNS_CNT_PER_WARP + COLUMNS_CNT_PER_WARP_BIG_ITER +
                                 (THREAD_ID_IN_WARP / task_split.rows_in_one_block * task_split.columns_per_thread) +
                                 j <
                             COLUMNS_CNT);
                        const size_t k = offset_local + j;
                        const float w = valid_index ? _weights[index] : 0.0f;
                        ws_->set(k, w);
                    }
                    offset_local += task_split.columns_per_thread;
                    lines_processed += task_split.rows_in_one_block;
                }
            }
        }
    }

    // Loads the feature “stripe” for our warp into a local array.
    // Takes into account a possible column tail (padding).
    __forceinline__ __device__ void load_feats(const float *__restrict__ _feats_global,
                                               float *__restrict__ _feats_local)
    {
#pragma unroll
        for (size_t i = 0; i < FEATS_CNT_PER_THREAD; ++i) {
            if constexpr (HAS_SMALL_WARP_ITERATIONS || HAS_PADDED_COLUMNS) {
                // Index within the warp “stripe” (with stride 32)
                size_t offset_in_warp = THREAD_ID_IN_WARP + THREADS_CNT_PER_WARP * i;
                // Global features vector index tied to our warp
                size_t index = WARP_ID_IN_BLOCK * COLUMNS_CNT_PER_WARP + offset_in_warp;
                bool valid_index = (offset_in_warp < COLUMNS_CNT_PER_WARP) && (index < COLUMNS_CNT);
                _feats_local[i] = valid_index ? _feats_global[index] : 0.0f;
            } else {
                // Fast path without checks (when there are no column tails)
                _feats_local[i] = _feats_global[WARP_ID_IN_BLOCK * COLUMNS_CNT_PER_WARP + THREAD_ID_IN_WARP +
                                                THREADS_CNT_PER_WARP * i];
            }
        }
    }

    /*
     * Main math: dot products of submatrix rows with the feature vector.
     * Results are reduced across warps in shared memory and emitted by the block’s “zero” warp.
     *
     * _accum_array — array of partial sums per row for a lane of the block’s “zero” warp.
     * _feats_local — local features loaded by load_feats.
     */
    __forceinline__ __device__ void calc(float *__restrict__ _accum_array, const float *__restrict__ _feats_local)
    {
// 1) Big row iterations (no tail)
#pragma unroll
        for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
            accum[i] = 0.f;
// All “full” 32-feature chunks (excluding the last one if there is a tail)
#pragma unroll
            for (size_t j = 0; j < (FEATS_CNT_PER_THREAD - 1); ++j) {
#pragma unroll
                for (size_t k = 0; k < THREADS_CNT_PER_WARP; ++k) {
                    // Distribute features across lanes via shuffle
                    float f = __shfl_sync(0, _feats_local[j], k);
                    // Locally loaded weight * corresponding feature
                    accum[i] += ws_->get(i * COLUMNS_CNT_PER_WARP + j * THREADS_CNT_PER_WARP + k) * f;
                }
            }

            // Last chunk: either full (32) or the column tail
            constexpr size_t last_warp_iter =
                (HAS_SMALL_WARP_ITERATIONS) ? COLUMNS_CNT_PER_WARP % THREADS_CNT_PER_WARP : THREADS_CNT_PER_WARP;
#pragma unroll
            for (size_t k = 0; k < last_warp_iter; ++k) {
                float f = __shfl_sync(0, _feats_local[FEATS_CNT_PER_THREAD - 1], k);
                accum[i] +=
                    ws_->get(i * COLUMNS_CNT_PER_WARP + (FEATS_CNT_PER_THREAD - 1) * THREADS_CNT_PER_WARP + k) * f;
            }
        }

        // 2) Tail rows
        if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
            size_t offset_local = BIG_ROW_ITER_CNT_PER_THREAD * COLUMNS_CNT_PER_WARP;
            // 2.1) p2-tasks for the “big” column part
            static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                constexpr size_t I = I_c;
                constexpr size_t n = p2_tasks[I];
                constexpr size_t threads_per_row = THREADS_CNT_PER_WARP / n;
                constexpr size_t columns_per_thread = COLUMNS_CNT_PER_WARP_BIG_ITER / threads_per_row;
                constexpr size_t BIG_FEAT_ITERS =
                    (HAS_SMALL_WARP_ITERATIONS) ? FEATS_CNT_PER_THREAD - 1 : FEATS_CNT_PER_THREAD;
                static_assert(COLUMNS_CNT_PER_WARP_BIG_ITER == THREADS_CNT_PER_WARP * BIG_FEAT_ITERS);
                static_assert(columns_per_thread == n * BIG_FEAT_ITERS);
                accum[BIG_ROW_ITER_CNT_PER_THREAD + I] = 0.f;
// All “full” feature chunks (no column tail)
#pragma unroll
                for (size_t j = 0; j < BIG_FEAT_ITERS; ++j) {
                    const size_t feat_offset = THREAD_ID_IN_WARP / n * n;
#pragma unroll
                    for (size_t k = 0; k < n; ++k) {
                        float f = __shfl_sync(0, _feats_local[j], k + feat_offset);
                        accum[BIG_ROW_ITER_CNT_PER_THREAD + I] += ws_->get(offset_local + j * n + k) * f;
                    }
                }
                offset_local += columns_per_thread;

                // Reduction across threads_per_row
                const uint8_t need_store = (THREAD_ID_IN_WARP < n);
#pragma unroll
                for (size_t j = 1; j < threads_per_row; ++j) {
                    float tmp = __shfl_sync(0, accum[BIG_ROW_ITER_CNT_PER_THREAD + I], THREAD_ID_IN_WARP % n + j * n);
                    accum[BIG_ROW_ITER_CNT_PER_THREAD + I] += need_store * tmp;
                }
            });

            // 2.2) Column tail in tail rows
            if constexpr (HAS_SMALL_BLOCK_ITERATIONS && HAS_SMALL_WARP_ITERATIONS) {
#pragma unroll
                for (size_t i = 0; i < task_split.cbi_cnt; ++i) {
                    constexpr size_t threads_per_row = THREADS_CNT_PER_WARP / task_split.rows_in_one_block;
                    accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + i] = 0.f;
// Only account for the tail column part
#pragma unroll
                    for (size_t j = 0; j < task_split.columns_per_thread; ++j) {
                        const size_t feat_offset =
                            THREAD_ID_IN_WARP / task_split.rows_in_one_block * task_split.columns_per_thread;
                        float f = __shfl_sync(0, _feats_local[FEATS_CNT_PER_THREAD - 1], j + feat_offset);
                        accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + i] += ws_->get(offset_local + j) * f;
                    }
                    offset_local += task_split.columns_per_thread;

                    // Reduction across threads_per_row
                    const uint8_t need_store = (THREAD_ID_IN_WARP < task_split.rows_in_one_block);
#pragma unroll
                    for (size_t j = 1; j < threads_per_row; ++j) {
                        float tmp = __shfl_sync(0, accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + i],
                                                THREAD_ID_IN_WARP % task_split.rows_in_one_block +
                                                    j * task_split.rows_in_one_block);
                        accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + i] += need_store * tmp;
                    }
                }
            }
        }

        // 3) Stitching across warps in shared memory
        __shared__ float partial_sum[WARPS_CNT * ROWS_CNT_PER_BLOCK];
// 3.1) Write “big” results to shared (each lane writes its own position)
#pragma unroll
        for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
            partial_sum[(WARP_ID_IN_BLOCK * THREADS_CNT_PER_WARP + THREAD_ID_IN_WARP) * BIG_ROW_ITER_CNT_PER_THREAD +
                        i] = accum[i];
        }

        // 3.2) Write tail rows to shared (if present)
        if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
            constexpr size_t big_iter_offset = WARPS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD;
            size_t lines_processed = 0;
            static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                constexpr size_t I = I_c;
                constexpr size_t n = p2_tasks[I];
                const bool need_store = THREAD_ID_IN_WARP < n;
                if (need_store) {
                    partial_sum[big_iter_offset + (lines_processed + THREAD_ID_IN_WARP) * WARPS_CNT +
                                WARP_ID_IN_BLOCK] = accum[BIG_ROW_ITER_CNT_PER_THREAD + I];
                }
                lines_processed += n;
            });

            // column tail
            if constexpr (HAS_SMALL_BLOCK_ITERATIONS && HAS_SMALL_WARP_ITERATIONS) {
                static_for<task_split.cbi_cnt>([&] __device__(auto I_c) {
                    constexpr size_t I = I_c;
                    constexpr size_t rows_to_process = std::min(
                        task_split.rows_in_one_block, TAIL_ROWS_CNT_PER_BLOCK - I * task_split.rows_in_one_block);
                    const uint8_t need_store = THREAD_ID_IN_WARP < rows_to_process;
                    partial_sum[big_iter_offset + (I * task_split.rows_in_one_block + THREAD_ID_IN_WARP) * WARPS_CNT +
                                WARP_ID_IN_BLOCK] += need_store * accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + I];
                });
            }
        }

        __syncthreads();

        // 4) Sum across warps and write into _accum_array on the block’s “zero” warp
        if (WARP_ID_IN_BLOCK == 0) {
// 4.1) Big rows
#pragma unroll
            for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
                _accum_array[i] = 0.f;
#pragma unroll
                for (size_t w_id = 0; w_id < WARPS_CNT; ++w_id) {
                    _accum_array[i] +=
                        partial_sum[(w_id * THREADS_CNT_PER_WARP + THREAD_ID_IN_WARP) * BIG_ROW_ITER_CNT_PER_THREAD +
                                    i];
                }
            }

            // 4.2) Tail rows
            if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
                size_t lines_processed = 0;
                static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                    constexpr size_t I = I_c;
                    constexpr size_t n = p2_tasks[I];
                    constexpr size_t big_iter_offset = WARPS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD;
                    const bool need_store = THREAD_ID_IN_WARP < n;
                    if (need_store) {
                        _accum_array[BIG_ROW_ITER_CNT_PER_THREAD + I] = 0.f;
                        for (size_t w_id = 0; w_id < WARPS_CNT; ++w_id) {
                            _accum_array[BIG_ROW_ITER_CNT_PER_THREAD + I] +=
                                partial_sum[big_iter_offset + (lines_processed + THREAD_ID_IN_WARP) * WARPS_CNT + w_id];
                        }
                    }
                    lines_processed += n;
                });
            }
        }
    }

    // Writes the final result from _accum_array into the output array _output_data,
    // adding the corresponding bias. Only the block’s “zero” warp writes.
    __forceinline__ __device__ void save_result(const float *__restrict__ _accum_array,
                                                float *__restrict__ _output_data)
    {
        if (WARP_ID_IN_BLOCK == 0) {
#pragma unroll
            for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
                _output_data[(BLOCK_ID * THREADS_CNT_PER_WARP + THREAD_ID_IN_WARP) * BIG_ROW_ITER_CNT_PER_THREAD + i] =
                    _accum_array[i] + bias_local[i];
            }

            if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
                size_t lines_processed = 0;
                static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                    constexpr size_t I = I_c;
                    constexpr size_t n = p2_tasks[I];
                    const size_t big_iter_offset = BLOCKS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD +
                                                   BLOCK_ID * TAIL_ROWS_CNT_PER_BLOCK;
                    const size_t row = big_iter_offset + lines_processed + THREAD_ID_IN_WARP;
                    if (THREAD_ID_IN_WARP < n && row < ROWS_CNT) {
                        _output_data[row] =
                            _accum_array[BIG_ROW_ITER_CNT_PER_THREAD + I] + bias_local[BIG_ROW_ITER_CNT_PER_THREAD + I];
                    }
                    lines_processed += n;
                });
            }
        }
    }

    // Same as save_result(), but applies an activation to (accum + bias) before storing.
    // Act must be a device-callable functor with signature: float operator()(float) const.
    template <class Act>
    __forceinline__ __device__ void save_result_act(const float *__restrict__ _accum_array,
                                                    float *__restrict__ _output_data, Act act)
    {
        if (WARP_ID_IN_BLOCK == 0) {
#pragma unroll
            for (size_t i = 0; i < BIG_ROW_ITER_CNT_PER_THREAD; ++i) {
                float v = _accum_array[i] + bias_local[i];
                _output_data[(BLOCK_ID * THREADS_CNT_PER_WARP + THREAD_ID_IN_WARP) * BIG_ROW_ITER_CNT_PER_THREAD + i] =
                    act(v);
            }

            if constexpr (HAS_SMALL_BLOCK_ITERATIONS) {
                size_t lines_processed = 0;
                static_for<P2_TASKS_CNT>([&] __device__(auto I_c) {
                    constexpr size_t I = I_c;
                    constexpr size_t n = p2_tasks[I];
                    const size_t big_iter_offset = BLOCKS_CNT * THREADS_CNT_PER_WARP * BIG_ROW_ITER_CNT_PER_THREAD +
                                                   BLOCK_ID * TAIL_ROWS_CNT_PER_BLOCK;
                    const size_t row = big_iter_offset + lines_processed + THREAD_ID_IN_WARP;
                    if (THREAD_ID_IN_WARP < n && row < ROWS_CNT) {
                        float v =
                            _accum_array[BIG_ROW_ITER_CNT_PER_THREAD + I] + bias_local[BIG_ROW_ITER_CNT_PER_THREAD + I];
                        _output_data[row] = act(v);
                    }
                    lines_processed += n;
                });
            }
        }
    }

private:
    Storage *ws_;
    const size_t BLOCK_ID;
    const size_t WARP_ID_IN_BLOCK;
    const size_t THREAD_ID_IN_WARP;
    float bias_local[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT];
    float accum[BIG_ROW_ITER_CNT_PER_THREAD + P2_TASKS_CNT + task_split.cbi_cnt];
};
