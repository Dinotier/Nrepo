cmake_minimum_required(VERSION 3.18)
project(compact_weights CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

add_executable(compact_weights_test compact_weights.cu)

target_compile_options(compact_weights_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas -v>)
