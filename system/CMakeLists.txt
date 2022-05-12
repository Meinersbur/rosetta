# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
#include(FetchContent)
#FetchContent_Declare(googlebenchmark
#    GIT_REPOSITORY https://github.com/google/benchmark.git
#)
#FetchContent_MakeAvailable(googlebenchmark)
add_subdirectory(benchmark)

# Rosetta system
add_subdirectory(rosetta)
