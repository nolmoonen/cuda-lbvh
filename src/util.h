// Copyright (c) 2022-2026 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

typedef unsigned int uint;
typedef unsigned char uchar;

template <
    typename T,
    typename U,
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__device__ __host__ constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

#define RETURN_IF_FALSE(call)                                                       \
    do {                                                                            \
        const bool res_ = call;                                                     \
        if (!res_) {                                                                \
            fprintf(stderr, #call "returned false at " __FILE__ ":%d\n", __LINE__); \
            return false;                                                           \
        }                                                                           \
    } while (0)

#define LOG_CUDA(err)                          \
    do {                                       \
        fprintf(                               \
            stderr,                            \
            "CUDA error '%s' (%d) at %s:%d\n", \
            cudaGetErrorString(err),           \
            static_cast<int>(err),             \
            __FILE__,                          \
            __LINE__);                         \
    } while (0)

#define RETURN_IF_CUDA_ERR(call)       \
    do {                               \
        const cudaError_t err_ = call; \
        if (err_ != cudaSuccess) {     \
            LOG_CUDA(err_);            \
            return false;              \
        }                              \
    } while (0)

template <typename t>
struct allocator_gpu {
    static t* malloc(size_t size)
    {
        t* ptr          = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess || ptr == nullptr) {
            LOG_CUDA(err);
            return nullptr;
        }

        return ptr;
    }

    static void free(t* ptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            LOG_CUDA(err);
        }
    }
};

template <typename t>
struct allocator_cpu {
    static t* malloc(size_t size)
    {
        t* ptr = static_cast<t*>(std::malloc(size));
        if (ptr == nullptr) {
            fprintf(stderr, "Failed to allocate\n");
        }
        return ptr;
    }

    static void free(t* ptr) { std::free(ptr); }
};

// Simple non-initializing buffer.
template <typename t, typename allocator>
class buffer {
  public:
    buffer() : ptr(nullptr), num_elements(0) {}
    ~buffer()
    {
        if (ptr != nullptr) {
            allocator::free(ptr);
            ptr          = nullptr;
            num_elements = 0;
        }
        assert(num_elements == 0);
    }

    buffer(const buffer&)            = delete;
    buffer(buffer&&)                 = delete;
    buffer& operator=(const buffer&) = delete;
    buffer& operator=(buffer&&)      = delete;

    t* get_ptr() noexcept { return ptr; }

    const t* get_ptr() const noexcept { return ptr; }

    int get_num_elements() const noexcept { return num_elements; }

    // Resizes the underlying allocation without copying
    // or initializing.
    [[nodiscard]] bool resize(int num_elem) noexcept
    {
        if (num_elem < 0) return false;
        if (num_elem <= num_elements) return true;

        if (ptr != nullptr) {
            allocator::free(ptr);
            ptr          = nullptr;
            num_elements = 0;
        }

        ptr = static_cast<t*>(allocator::malloc(num_elem * sizeof(t)));
        if (ptr == nullptr) {
            return false;
        }

        num_elements = num_elem;

        return true;
    }

  private:
    t* ptr;
    int num_elements;
};

// Simple non-initializing device buffer.
template <typename t>
using buf_gpu = buffer<t, allocator_gpu<t>>;

// Simple non-initializing device buffer.
template <typename t>
using buf_cpu = buffer<t, allocator_cpu<t>>;
