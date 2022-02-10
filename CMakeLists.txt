cmake_minimum_required(VERSION 3.21.3)
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )
 
project(rosetta
  VERSION 0.0.1
  LANGUAGES C CXX
)



include(CheckLanguage)



## Configuration ###############################################################


set(OPENMP_OFFLOADING_CFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70"
  CACHE STRING "Compiler arguments for OpenMP offloading")
set(OPENMP_OFFLOADING_LDFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70;-lomptarget"
  CACHE STRING "Linker arguments for OpenMP offloading")
set(CMAKE_CUDA_ARCHITECTURES OFF) # ???


## Dependencies ################################################################

# Python
find_package(Python 3.6.13 REQUIRED)

# pthreads
#find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# OpenMP offloading
find_package(OpenMPOffload REQUIRED)

# Google Benchmark
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)

# CUDA
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif ()


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



## Compiler-preconfigurations ##################################################

option(XCELLENT_ENABLE_SERIAL ON)

set(_cuda_enabled_default OFF)
if (CMAKE_CUDA_COMPILER AND CMAKE_CUDA_HOST_COMPILER)
    set(_cuda_enabled_default ON)
endif ()
option(XCELLENT_ENABLE_CUDA ${_cuda_enabled_default})


option(XCELLENT_ENABLE_OPENMP_PARALLEL ON)
option(XCELLENT_ENABLE_OPENMP_TASKS ON)
option(XCELLENT_ENABLE_OPENMP_TARGET ON)



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



