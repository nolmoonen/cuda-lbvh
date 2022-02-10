set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/trace.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/build.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cub_helper.cu
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

OPTIX_add_sample_executable(cuda-lbvh target_name
        build.cu
        build.h
        bvh.h
        camera.h
        cub_helper.cu
        cub_helper.h
        cuda_check.h
        main.cpp
        obj_reader.cpp
        obj_reader.h
        trace.cu
        trace.h
        util.cuh
        util.h
        vec_math_helper.h)

target_link_libraries(${target_name} ${CUDA_LIBRARIES} sutil_7_sdk stb)

# set the binary and scene directories for the benchmark script
set(BINARY_DIR "${CMAKE_BINARY_DIR}")
set(SCENE_DIR "${CMAKE_SOURCE_DIR}/scenes")
configure_file(
        ${CMAKE_SOURCE_DIR}/python/benchmark.py.in
        ${CMAKE_BINARY_DIR}/benchmark/benchmark.py @ONLY)