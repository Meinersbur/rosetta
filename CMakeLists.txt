cmake_minimum_required(VERSION 3.21.3)
project(rosetta
  VERSION 0.0.1
  LANGUAGES CXX
)

list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

include(FetchContent)
include(CheckLanguage)


## Configuration ###############################################################

set(OPENMP_OFFLOADING_CFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70"
  CACHE STRING "Compiler arguments for OpenMP offloading")
set(OPENMP_OFFLOADING_LDFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70;-lomptarget"
  CACHE STRING "Linker arguments for OpenMP offloading")
set(CMAKE_CUDA_ARCHITECTURES OFF) # ???


## Dependencies ################################################################

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)

# Python
find_package(Python 3.6.13 REQUIRED)

# CUDA
if (NOT DEFINED ROSETTA_ENABLE_CUDA OR ROSETTA_ENABLE_CUDA)
  message("Looking for CUDA...")
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
  message("Enable CUDA...")
    enable_language(CUDA)
  elseif (ROSETTA_ENABLE_CUDA)
    message(FATAL_ERROR "ROSETTA_ENABLE_CUDA enabled but CUDA not found")
  endif ()
endif ()

# pthreads
#find_package(Threads REQUIRED)

if (NOT DEFINED ROSETTA_ENABLE_OPENMP_PARALLEL OR ROSETTA_ENABLE_OPENMP_PARALLEL)
  # OpenMP
  if (ROSETTA_ENABLE_OPENMP_PARALLEL)
    find_package(OpenMP REQUIRED)
  else ()
    find_package(OpenMP)
  endif ()

  # OpenMP offloading
  find_package(OpenMPOffload)
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

option(ROSETTA_ENABLE_SERIAL "Include PPM: non-parallel" ON)

set(_cuda_enabled_default OFF)
if (CMAKE_CUDA_COMPILER)
  message("Enable CUDA by default...")
  set(_cuda_enabled_default ON)
endif ()
message("_cuda_enabled_default: ${_cuda_enabled_default}")
message("ROSETTA_ENABLE_CUDA: ${ROSETTA_ENABLE_CUDA}")
option(ROSETTA_ENABLE_CUDA "Include PPM: CUDA" ${_cuda_enabled_default})
message("ROSETTA_ENABLE_CUDA: ${ROSETTA_ENABLE_CUDA}")

set(_openmp_paralel_enabled_default OFF)
if (OpenMP_FOUND)
  set(_openmp_paralel_enabled_default ON)
endif ()
option(ROSETTA_ENABLE_OPENMP_PARALLEL "Include PPM: #pragma omp parallel" ON)

option(ROSETTA_ENABLE_OPENMP_TASKS "Include PPM: #pragma omp task" OFF)
option(ROSETTA_ENABLE_OPENMP_TARGET "Include PPM: #pragma omp target" OFF)



## Build #######################################################################

set_property(GLOBAL PROPERTY benchmarks_serial)
set_property(GLOBAL PROPERTY benchmarks_cuda)
set_property(GLOBAL PROPERTY benchmarks_omp_parallel)

add_custom_target(gbench-serial)
add_custom_target(gbench-cuda)
add_custom_target(gbench-omp_parallel)

add_custom_target(gbench)
add_dependencies(gbench gbench-serial gbench-cuda gbench-omp_parallel)

add_subdirectory(src)

get_property(_tmp GLOBAL PROPERTY benchmarks_cuda)
message(VERBOSE "CUDA benchmarks: ${_tmp}")


## Run #########################################################################


set(RUNNER_EXECUTABLE "${CMAKE_SOURCE_DIR}/runner.py")

set(_benchmarks)
set(_benchmarks_run "#! /usr/bin/env bash
${Python_EXECUTABLE} ${RUNNER_EXECUTABLE}")


get_property(_benchmarks_cuda GLOBAL PROPERTY benchmarks_cuda)
list(APPEND _benchmarks ${_benchmarks_cuda})


foreach (_target IN LISTS _benchmarks_cuda)
  set(_benchmarks_run "${_benchmarks_run} \\\n  --cuda-exe $<TARGET_FILE:${_target}>")
endforeach ()

file(GENERATE 
    OUTPUT "${CMAKE_BINARY_DIR}/run-$<CONFIG>.sh"
    CONTENT "${_benchmarks_run}"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
#file(CHMOD "${CMAKE_BINARY_DIR}/run.sh" FILE_PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
# file (COPY ${CMAKE_BINARY_DIR}/tmp/somescript.sh DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)

execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "run-${CMAKE_DEFAULT_BUILD_TYPE}.sh" "run.sh" 
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      )


  
add_custom_target(run
    DEPENDS gbench
    COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--exedir" "${CMAKE_CURRENT_BINARY_DIR}/src" 
    COMMENT "Run benchmarks ..."
    USES_TERMINAL VERBATIM
)

add_custom_target(run-cuda
    DEPENDS gbench-cuda
    COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" ${_benchmarks_cuda_exe}
    COMMENT "Run CUDA benchmarks ..."
    USES_TERMINAL VERBATIM
)

## Summary #########################################################################

set(_disabled_ppm)


message("")
message("Configuration summary")
message("---------------------")
message("Enabled PPMs:")
if (ROSETTA_ENABLE_SERIAL)
  message("  Serial: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
  set(_disabled_ppm "Serial ${_disabled_ppm}")
endif ()
if (ROSETTA_ENABLE_CUDA)
  message("  CUDA: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION})")
else ()
  set(_disabled_ppm "CUDA ${_disabled_ppm}")
endif ()
if (ROSETTA_ENABLE_OPENMP_PARALLEL)
  message("  OpenMP-Parallel: ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
  set(_disabled_ppm "OpenMP-parallel ${_disabled_ppm}")
endif ()
if (_disabled_ppm)
message("Disabled PPMs: ${_disabled_ppm}")
else ()
message("Disabled PPMs: -none-")
endif ()
message("")

