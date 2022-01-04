cmake_minimum_required(VERSION 3.22.1)
project(rosetta
  VERSION 0.0.1
  LANGUAGES C CXX CUDA
)

## Configuration ###############################################################


## Dependencies ################################################################

# pthreads
find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# Google Benchmark
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)


## Introspection ################################################################

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


## Build #######################################################################

add_subdirectory(src)


## Run #########################################################################

