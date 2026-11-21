cmake_minimum_required(VERSION 3.18)

# 61 = Pascal (GTX 1080 Ti).  Must be set before project() so the compiler
# default does not win.  Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89 90)
endif()

project(compact_weights CUDA CXX)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "CUDA 12.x required, found ${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND "61" IN_LIST CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot build sm_61 (GTX 1080 Ti); "
        "use CUDA 12.x or drop 61 from CMAKE_CUDA_ARCHITECTURES")
endif()

set(CMAKE_CUDA_STANDARD 17)

add_executable(compact_weights_test compact_weights.cu)

target_compile_options(compact_weights_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas -v>)
