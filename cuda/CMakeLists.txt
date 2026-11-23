set(FIVER_LIB_SOURCES
    src/forward_gemv.cu
    src/forward_gemm.cu
    src/forward_int8.cu
    src/backward.cu
    src/api.cu)
set(FIVER_CLI_SOURCES
    tools/cli.cu
    tests/tests.cu
    bench/bench.cu)

if(FIVER_EMU)
    # CPU emulator: the same .cu sources compiled as C++20 with threads.
    set_source_files_properties(${FIVER_LIB_SOURCES} ${FIVER_CLI_SOURCES} PROPERTIES LANGUAGE CXX)
    find_package(Threads REQUIRED)
endif()

add_library(fiver_cuda SHARED ${FIVER_LIB_SOURCES})
target_include_directories(fiver_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(compact_weights_test ${FIVER_CLI_SOURCES})
target_link_libraries(compact_weights_test PRIVATE fiver_cuda)
# The C# controller expects build/compact_weights_test.
set_target_properties(compact_weights_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# int8 table decodes: shuffle + shift/or versus shuffle + PRMT (correctness test)
add_executable(prmt_lut_test bench/prmt_lut.cu)
target_include_directories(prmt_lut_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(prmt_lut_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(FIVER_EMU)
    set_source_files_properties(bench/prmt_lut.cu PROPERTIES LANGUAGE CXX)
endif()

foreach(t fiver_cuda compact_weights_test prmt_lut_test)
    if(FIVER_EMU)
        target_compile_definitions(${t} PRIVATE FIVER_EMU)
        target_compile_features(${t} PRIVATE cxx_std_20)
        target_compile_options(${t} PRIVATE -x c++ -O2 -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable)
        target_link_libraries(${t} PRIVATE Threads::Threads)
    else()
        target_compile_options(${t} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda -Xptxas -v>)
    endif()
endforeach()
