cmake_minimum_required(VERSION 3.18)

# FIVER_EMU=ON builds everything for the CPU emulator (no CUDA toolkit needed):
#   cmake -S . -B build-emu -DFIVER_EMU=ON && cmake --build build-emu
option(FIVER_EMU "Build the CUDA sources for the CPU emulator (tests without a GPU)" OFF)

# 61 = Pascal (GTX 1080 Ti).  Must be set before project() so the compiler
# default does not win.  Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89 90)
endif()

if(FIVER_EMU)
    project(compact_weights LANGUAGES CXX)
else()
    project(compact_weights LANGUAGES CUDA CXX)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR "CUDA 12.x required, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND "61" IN_LIST CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot build sm_61 (GTX 1080 Ti); "
            "use CUDA 12.x or drop 61 from CMAKE_CUDA_ARCHITECTURES")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
endif()

add_subdirectory(cuda)
