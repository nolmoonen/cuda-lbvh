#pragma once

#include <cuda_runtime.h>

__forceinline__ __device__ unsigned int get_block_id()
{
    return blockIdx.x + blockIdx.y * gridDim.x +
           gridDim.x * gridDim.y * blockIdx.z;
}

__forceinline__ __device__ unsigned int get_thread_id(unsigned int block_id)
{
    return block_id * (blockDim.x * blockDim.y * blockDim.z)
           + (threadIdx.z * (blockDim.x * blockDim.y))
           + (threadIdx.y * blockDim.x) + threadIdx.x;
}
