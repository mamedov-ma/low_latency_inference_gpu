#pragma once

#include <tuple>
#include <type_traits>
#include <assert.h>

template <size_t N, typename T = float>
class RegStor {
public:
    static_assert(N > 0, "RegStor<N>: N must be greater than zero");
    using value_type = T;
    static constexpr size_t size = N;

    __device__ __forceinline__ RegStor() = default;

    __device__ __forceinline__ T get(size_t i) const
    {
#ifndef NDEBUG
        assert(i < N);
#endif
        return data_[i];
    }
    __device__ __forceinline__ void set(size_t i, T v)
    {
#ifndef NDEBUG
        assert(i < N);
#endif
        data_[i] = v;
    }

    __device__ __forceinline__ constexpr size_t count() const
    {
        return N;
    }

private:
    T data_[N];
};

template <size_t N, typename T = float>
class ShmemStor {
public:
    static_assert(N > 0, "ShmemStor<N>: N must be greater than zero");
    using value_type = T;
    static constexpr size_t size = N;

    __device__ __forceinline__ explicit ShmemStor(T *base = nullptr) : ptr_(base) {}

    __device__ __forceinline__ T get(size_t i) const
    {
#ifndef NDEBUG
        assert(ptr_ != nullptr && i < N);
#endif
        return ptr_[i];
    }
    __device__ __forceinline__ void set(size_t i, T v)
    {
#ifndef NDEBUG
        assert(ptr_ != nullptr && i < N);
#endif
        ptr_[i] = v;
    }

    __device__ __forceinline__ constexpr size_t count() const
    {
        return N;
    }
    __device__ __forceinline__ T *data() const
    {
        return ptr_;
    }

    static constexpr size_t bytes_per_thread()
    {
        return N * sizeof(T);
    }

private:
    T *ptr_;
};

template <size_t N, typename T = float>
class GlobMemStor {
public:
    static_assert(N > 0, "GlobMemStor<N>: N must be greater than zero");
    using value_type = T;
    static constexpr size_t size = N;

    __device__ __forceinline__ explicit GlobMemStor(T *base = nullptr) : ptr_(base) {}

    __device__ __forceinline__ T get(size_t i) const
    {
#ifndef NDEBUG
        assert(ptr_ != nullptr && i < N);
#endif
        return ptr_[i];
    }
    __device__ __forceinline__ void set(size_t i, T v)
    {
#ifndef NDEBUG
        assert(ptr_ != nullptr && i < N);
#endif
        ptr_[i] = v;
    }

    __device__ __forceinline__ constexpr size_t count() const
    {
        return N;
    }
    __device__ __forceinline__ T *data() const
    {
        return ptr_;
    }

private:
    T *ptr_;
};

template <typename... Storages>
class WeightsStorage {
public:
    static_assert(sizeof...(Storages) >= 1, "WeightsStorage must have at least one storage unit");

    using value_type = typename std::tuple_element<0, std::tuple<Storages...>>::type::value_type;

    static constexpr size_t total_count = (Storages::size + ...);

    __device__ __forceinline__ explicit WeightsStorage(Storages... ss) : storages_(ss...) {}

    __device__ __forceinline__ value_type get(size_t idx) const
    {
#ifndef NDEBUG
        assert(idx < total_count);
#endif
        return get_impl<0>(idx);
    }

    __device__ __forceinline__ void set(size_t idx, value_type v)
    {
#ifndef NDEBUG
        assert(idx < total_count);
#endif
        set_impl<0>(idx, v);
    }

    template <size_t I>
    __device__ __forceinline__ auto &storage()
    {
        return std::get<I>(storages_);
    }
    template <size_t I>
    __device__ __forceinline__ const auto &storage() const
    {
        return std::get<I>(storages_);
    }

private:
    std::tuple<Storages...> storages_;

    template <size_t I>
    __device__ __forceinline__ value_type get_impl(size_t idx) const
    {
        if constexpr (I >= sizeof...(Storages)) {
            return value_type(0);
        } else {
            const auto &s = std::get<I>(storages_);
            const size_t n = s.count();
            return (idx < n) ? s.get(idx) : get_impl<I + 1>(idx - n);
        }
    }

    template <size_t I>
    __device__ __forceinline__ void set_impl(size_t idx, value_type v)
    {
        if constexpr (I >= sizeof...(Storages)) {
            return;
        } else {
            auto &s = std::get<I>(storages_);
            const size_t n = s.count();
            if (idx < n) {
                s.set(idx, v);
            } else {
                set_impl<I + 1>(idx - n, v);
            }
        }
    }
};

template <size_t REG_N, size_t SHMEM_N, size_t GLOB_N>
__device__ auto make_weights_storage(float *shmem = nullptr, float *gmem = nullptr)
{
    static_assert(REG_N > 0 || SHMEM_N > 0 || GLOB_N > 0, "Empty storage");

    if constexpr (REG_N > 0 && SHMEM_N > 0 && GLOB_N > 0) {
        using Reg = RegStor<REG_N, float>;
        using Shm = ShmemStor<SHMEM_N, float>;
        using Glob = GlobMemStor<GLOB_N, float>;
        using WS = WeightsStorage<Reg, Shm, Glob>;
        Reg r {};
        Shm s {shmem};
        Glob g {gmem};
        return WS {r, s, g};
    } else if constexpr (REG_N > 0 && SHMEM_N > 0) {
        using Reg = RegStor<REG_N, float>;
        using Shm = ShmemStor<SHMEM_N, float>;
        using WS = WeightsStorage<Reg, Shm>;
        Reg r {};
        Shm s {shmem};
        return WS {r, s};
    } else if constexpr (SHMEM_N > 0 && GLOB_N > 0) {
        using Shm = ShmemStor<SHMEM_N, float>;
        using Glob = GlobMemStor<GLOB_N, float>;
        using WS = WeightsStorage<Shm, Glob>;
        Shm s {shmem};
        Glob g {gmem};
        return WS {s, g};
    } else if constexpr (REG_N > 0 && GLOB_N > 0) {
        using Reg = RegStor<REG_N, float>;
        using Glob = GlobMemStor<GLOB_N, float>;
        using WS = WeightsStorage<Reg, Glob>;
        Reg r {};
        Glob g {gmem};
        return WS {r, g};
    } else if constexpr (REG_N > 0) {
        using Reg = RegStor<REG_N, float>;
        using WS = WeightsStorage<Reg>;
        Reg r {};
        return WS {r};
    } else if constexpr (SHMEM_N > 0) {
        using Shm = ShmemStor<SHMEM_N, float>;
        using WS = WeightsStorage<Shm>;
        Shm s {shmem};
        return WS {s};
    } else if constexpr (GLOB_N > 0) {
        using Glob = GlobMemStor<GLOB_N, float>;
        using WS = WeightsStorage<Glob>;
        Glob g {gmem};
        return WS {g};
    }
}
