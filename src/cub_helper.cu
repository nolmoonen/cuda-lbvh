#include <cub/device/device_radix_sort.cuh>
#include "cub_helper.h"
#include "cuda_check.h"

void radix_sort(
        int num_items,
        unsigned int *d_keys_in, unsigned int *d_keys_out,
        unsigned int *d_values_in, unsigned int *d_values_out)
{
    // determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // free temporary storage
    CUDA_CHECK(cudaFree(d_temp_storage));
}