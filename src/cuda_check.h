#pragma once

#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            fprintf(                                                           \
                stderr,                                                        \
                "CUDA call (" #call ") failed with error: '%s' (%s:%d)\n",     \
                cudaGetErrorString(error), __FILE__, __LINE__);                \
        }                                                                      \
    } while(0)


#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            fprintf(                                                           \
                stderr,                                                        \
                "CUDA error on synchronize with error '%s' (%s:%d)\n",         \
                cudaGetErrorString(error), __FILE__, __LINE__);                \
        }                                                                      \
    } while(0)