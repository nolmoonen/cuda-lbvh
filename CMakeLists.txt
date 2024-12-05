# Copyright (c) 2022-2024 Nol Moonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 3.17 FindCUDAToolkit
# 3.18 CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18)
project(jpeggpu LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_subdirectory(deps)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_executable(cuda-lbvh
    ${SOURCE_DIR}/build.cu
    ${SOURCE_DIR}/cub_helper.cu
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/obj_reader.cpp
    ${SOURCE_DIR}/trace.cu)
target_link_libraries(cuda-lbvh PUBLIC CUDA::cudart_static stb sutil_7_sdk)

# set the binary and scene directories for the benchmark script
set(BINARY_DIR "${CMAKE_BINARY_DIR}")
set(SCENE_DIR "${CMAKE_SOURCE_DIR}/scenes")
configure_file(
        ${CMAKE_SOURCE_DIR}/python/benchmark.py.in
        ${CMAKE_BINARY_DIR}/benchmark/benchmark.py @ONLY)
