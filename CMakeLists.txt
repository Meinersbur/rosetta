cmake_minimum_required(VERSION 3.22.1)
project(rosetta
  VERSION 0.0.1
  LANGUAGES C CXX CUDA
)

## Configuration ###############################################################

set(OPENMP_OFFLOADING_CFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70" CACHE STRING "Compiler arguments for OpenMP offloading")
set(OPENMP_OFFLOADING_LDFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70;-lomptarget" CACHE STRING "Linker arguments for OpenMP offloading")
set(CMAKE_CUDA_ARCHITECTURES OFF)

## Dependencies ################################################################

# pthreads
#find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# Google Benchmark
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)

# Python
find_package(Python 3.8.10 REQUIRED)


## Introspection ################################################################

# pointer restrict
check_cxx_source_compiles("
        int f(void *restrict x);
        int main(void) {return 0;}
    "
    HAVE_CXX_RESTRICT
)

check_cxx_source_compiles("
        int f(void *restrict x);
        int main(void) {return 0;}
    "
    HAVE_CXX___RESTRICT
)

# OpenMP offloading
add_library(OpenMP::OpenMP_Offload_CXX INTERFACE IMPORTED)
set_property(TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OPENMP_OFFLOADING_CFLAGS})
set_property(TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OPENMP_OFFLOADING_LDFLAGS})

## Build #######################################################################

add_custom_target(gbench)

add_subdirectory(src)


## Run #########################################################################

set(RUNNER_EXECUTABLE "${CMAKE_SOURCE_DIR}/runner.py")
add_custom_target(run
    DEPENDS gbench
    COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--exedir" "${CMAKE_CURRENT_BINARY_DIR}/src"
    COMMENT "Run benchmarks ..."
    USES_TERMINAL VERBATIM
)



