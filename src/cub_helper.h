#pragma once

#include <cstdint>

/// https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
void radix_sort(
        int num_items,
        unsigned int *d_keys_in, unsigned int *d_keys_out,
        unsigned int *d_values_in, unsigned int *d_values_out);
