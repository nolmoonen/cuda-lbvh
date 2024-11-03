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
