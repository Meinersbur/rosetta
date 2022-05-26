cmake_minimum_required(VERSION 3.17) # CMAKE_CROSS_CONFIGS
project(rosetta
  VERSION 0.0.1
  LANGUAGES CXX
)

list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/rosetta/cmake/modules"
  )

include(CheckLanguage)
include(CheckCXXSourceCompiles)



## Dependencies ################################################################

set(CMAKE_CXX_STANDARD 17)

# Python
find_package(Python 3.6.13 REQUIRED)

# Google Benchmark
#set(BENCHMARK_ENABLE_TESTING OFF)
#include(FetchContent)
#FetchContent_Declare(googlebenchmark
#    GIT_REPOSITORY https://github.com/google/benchmark.git
#)
#FetchContent_MakeAvailable(googlebenchmark)


# CUDA
if (NOT DEFINED ROSETTA_PPM_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES OFF) # ???
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  endif ()
elseif (ROSETTA_PPM_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES OFF) # ???
  check_language(CUDA)
  enable_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA enabled but no CUDA not found")
  endif ()
endif ()

# For nvidia platform
find_package(CUDAToolkit)


# OpenCL
if (NOT DEFINED ROSETTA_PPM_OPENCL)
  find_package(OpenCL)
elseif (ROSETTA_PPM_OPENCL)
  find_package(OpenCL REQUIRED)
endif ()


# pthreads
#find_package(Threads REQUIRED)


# OpenMP
if (NOT DEFINED ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_PARALLEL
    OR NOT DEFINED ROSETTA_PPM_OPENMP_TASK OR ROSETTA_PPM_OPENMP_TASK
    OR NOT DEFINED ROSETTA_PPM_OPENMP_TARGET OR ROSETTA_PPM_OPENMP_TARGET)
  if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK)
    find_package(OpenMP REQUIRED)
  else ()
    find_package(OpenMP)
  endif ()

  # OpenMP offloading
  if (NOT DEFINED ROSETTA_PPM_OPENMP_TARGET OR ROSETTA_PPM_OPENMP_TARGET)
    if (ROSETTA_PPM_OPENMP_TARGET)
      find_package(OpenMPOffload REQUIRED)
    else ()
      find_package(OpenMPOffload REQUIRED)
    endif ()
  endif()
endif ()


# OpenACC
if (NOT DEFINED ROSETTA_PPM_OPENACC)
  find_package(OpenACC)
elseif (ROSETTA_PPM_OPENACC)
  find_package(OpenACC REQUIRED)
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
        int f(void *__restrict x);
        int main(void) {return 0;}
    "
    HAVE_CXX___RESTRICT
)
if (HAVE_CXX_RESTRICT)
  set (KEYWORD_RESTRICT "restrict")
elif (HAVE_CXX___RESTRICT)
  set (KEYWORD_RESTRICT "__restrict")
else ()
  set (KEYWORD_RESTRICT "")
endif ()



## Configuration ###############################################################

option(ROSETTA_PPM_SERIAL "Enable serial reference" ON)

set(_cuda_enabled_default OFF)
if (CMAKE_CUDA_COMPILER)
  set(_cuda_enabled_default ON)
endif ()
option(ROSETTA_PPM_CUDA "Enable CUDA PPM" ${_cuda_enabled_default})

set(ROSETTA_PLATFORM_NVIDIA OFF)
if (TARGET CUDA::cudart)
  set(ROSETTA_PLATFORM_NVIDIA ON)
endif ()

set(_opencl_enabled_default OFF)
if (OpenCL_FOUND)
    set(_opencl_enabled_default ON)
endif ()
option(ROSETTA_PPM_OPENCL "Enable OpenCL PPM" ${_opencl_enabled_default})

set(_openmp_thread_enabled_default OFF)
if (OpenMP_CXX_FOUND)
  set(_openmp_thread_enabled_default ON)
endif ()
option(ROSETTA_PPM_OPENMP_PARALLEL "Enable OpenMP '#pragma omp parallel' PPM" ${_openmp_thread_enabled_default})
option(ROSETTA_PPM_OPENMP_TASK "Enable OpenMP '#pragma omp task' PPM" ${_openmp_thread_enabled_default})

set(_openmp_target_enabled_default OFF)
if (OpenMP_CXX_FOUND AND OpenMPOffload_FOUND)
  set(_openmp_target_enabled_default ON)
endif ()
option(ROSETTA_PPM_OPENMP_TARGET "Enable OpenMP '#pragma omp target' PPM" ${_openmp_target_enabled_default})

set(_openacc_enabled_default OFF)
if (OpenACC_CXX_FOUND)
  set(_openacc_enabled_default ON)
endif ()
option(ROSETTA_PPM_OPENACC "Enable OpenACC PPM" ${_openacc_enabled_default})




## Compiler-preconfigurations ##################################################




## Build #######################################################################

# Library ----------------------------------------------------------------------

add_subdirectory(rosetta)


# Benchmarks -------------------------------------------------------------------

set_property(GLOBAL PROPERTY benchmarks)
set_property(GLOBAL PROPERTY benchmarks_serial)
set_property(GLOBAL PROPERTY benchmarks_cuda)
set_property(GLOBAL PROPERTY benchmarks_openmp_parallel)
set_property(GLOBAL PROPERTY benchmarks_openmp_task)
set_property(GLOBAL PROPERTY benchmarks_openmp_target)

add_custom_target(build)

if (ROSETTA_PPM_SERIAL)
  add_custom_target(build-serial)
  add_dependencies(build build-serial)
endif ()
if (ROSETTA_PPM_CUDA)
  add_custom_target(build-cuda)
  add_dependencies(build build-cuda)
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK OR ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp)
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  add_custom_target(build-openmp-parallel)
  add_dependencies(build build-openmp-parallel)
  add_dependencies(build-openmp build-openmp-parallel)
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  add_custom_target(build-openmp-task)
  add_dependencies(build build-openmp-task)
  add_dependencies(build-openmp build-openmp-task)
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp-target)
  add_dependencies(build build-openmp-target)
  add_dependencies(build-openmp build-openmp-target)
endif ()


add_subdirectory(benchmarks)


## Run #########################################################################


set(RUNNER_EXECUTABLE "${CMAKE_SOURCE_DIR}/runner.py")

if (CMAKE_CONFIGURATION_TYPES)
  set(_build_types ${CMAKE_CONFIGURATION_TYPES})
elseif (CMAKE_BUILD_TYPE)
  set(_build_types "${CMAKE_BUILD_TYPE}")
else ()
  message(FATAL_ERROR "Cannot determine the build types")
endif ()

message("_build_types : ${_build_types}")
if (CMAKE_BUILD_TYPE)
  set(_default_build_type "${CMAKE_BUILD_TYPE}")
elseif (CMAKE_DEFAULT_BUILD_TYPE) # Only Ninja Multi-Config Generator
  set(_default_build_type "${CMAKE_DEFAULT_BUILD_TYPE}")
elseif ("Release" IN_LIST _build_types)
  set(_default_build_type "Release")
  # By default, Debug is first in CMAKE_CONFIGURATION_TYPES and becomes the implicit default. But we want Release to be the default by default.
  set(CMAKE_DEFAULT_BUILD_TYPE "Release")
elseif (CMAKE_CONFIGURATION_TYPES)
  list(GET CMAKE_CONFIGURATION_TYPES 0 _default_build_type)
else ()
  set(_default_build_type)
endif ()
message("CMAKE_DEFAULT_BUILD_TYPE: ${CMAKE_DEFAULT_BUILD_TYPE}")
message("__default_build_type: ${_default_build_type}")





# TODO: allow use as config file; run only with __name__ = '__main__'
set(_benchmarks)
set(_benchmarks_run "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')
from rosetta import run_benchs

")


get_property(_benchmarks_serial GLOBAL PROPERTY benchmarks_serial)
list(APPEND _benchmarks ${_benchmarks_serial})
get_property(_benchmarks_cuda GLOBAL PROPERTY benchmarks_cuda)
list(APPEND _benchmarks ${_benchmarks_cuda})
get_property(_benchmarks_openmp_parallel GLOBAL PROPERTY benchmarks_openmp_parallel)
list(APPEND _benchmarks ${_benchmarks_openmp_parallel})
get_property(_benchmarks_openmp_task GLOBAL PROPERTY benchmarks_openmp_task)
list(APPEND _benchmarks ${_benchmarks_openmp_task})
get_property(_benchmarks_openmp_target GLOBAL PROPERTY benchmarks_openmp_target)
list(APPEND _benchmarks ${_benchmarks_openmp_target})

set(_benchmarks_run_arglist)


function (enable_ppm name)
  cmake_parse_arguments(_arg "" "" "TARGETS"  ${ARGN} )

  set(_benchmarks_run)
  foreach (_target IN LISTS _arg_TARGETS)
    list(APPEND _benchmarks_run "\"$<TARGET_FILE:${_target}>\"")
  endforeach ()
  list(JOIN _benchmarks_run ",\n  " _benchmarks_pylist)
  set(_benchmarks_pylist "${name}=[\n  ${_benchmarks_pylist}]")

  # Add to list that runs all benchmarks
  list(APPEND _benchmarks_run_arglist "${_benchmarks_pylist}")
  set(_benchmarks_run_arglist "${_benchmarks_run_arglist}" PARENT_SCOPE)

  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/run-${name}-$<CONFIG>.py"
    CONTENT "${_benchmarks_run}run_benchs(config='$<CONFIG>',${_benchmarks_pylist})"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
endfunction()



if (ROSETTA_PPM_SERIAL)
  enable_ppm(serial TARGETS ${_benchmarks_serial})
endif ()

if (ROSETTA_PPM_CUDA)
  enable_ppm(cuda TARGETS ${_benchmarks_cuda})
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL)
  enable_ppm(omp_parallel TARGETS ${_benchmarks_openmp_parallel})
endif ()

if (ROSETTA_PPM_OPENMP_TASK)
  enable_ppm(omp_task TARGETS ${_benchmarks_openmp_task})
endif ()

if (ROSETTA_PPM_OPENMP_TARGET)
  enable_ppm(omp_target TARGETS ${_benchmarks_openmp_target})
endif ()




list(JOIN _benchmarks_run_arglist ",\n  " _benchmarks_run_pyarglist)
file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/run-$<CONFIG>.py"
    CONTENT "${_benchmarks_run}run_benchs(config='$<CONFIG>',${_benchmarks_run_pyarglist})"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )



foreach (_conftype IN LISTS _build_types)
  add_custom_target(run-${_conftype}
    DEPENDS gbench
    COMMAND  Python::Interpreter "run-${_conftype}.py"
    COMMENT "Run benchmarks for ${_conftype} ..."
    USES_TERMINAL VERBATIM
  )
endforeach ()



if (_default_build_type)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "run-${_default_build_type}.py" "run.py"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_custom_target(run
    COMMAND Python::Interpreter "run.py"
    COMMENT "Run all benchmarks ..."
    USES_TERMINAL VERBATIM
  )
  add_dependencies(run build)

  if (ROSETTA_PPM_SERIAL)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "run-serial-${_default_build_type}.py" "run-serial.py"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_custom_target(run-serial
      COMMAND Python::Interpreter "run-serial.py"
      COMMENT "Run serial benchmarks ..."
      USES_TERMINAL VERBATIM
    )
    add_dependencies(run-serial build-serial)
  endif ()

  if (ROSETTA_PPM_CUDA)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "run-cuda-${_default_build_type}.py" "run-cuda.py"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_custom_target(run-cuda
      COMMAND Python::Interpreter "run-cuda.py"
      COMMENT "Run CUDA benchmarks ..."
      USES_TERMINAL VERBATIM
    )
    add_dependencies(run-cuda build-cuda)
  endif ()
endif ()






## Summary #########################################################################

set(_disabled_ppm)
set(_disabled_platform)

set(_extra_flags)
if (ROSETTA_NO_EXCEPTIONS)
  set(_extra_flags "-fno-exceptions")
endif ()

message("")
message("Configuration Summary")
message("---------------------")
if (CMAKE_BUILD_TYPE)
  message("Build Configuration: ${CMAKE_BUILD_TYPE}")
else ()
  message("Build Configurations: ${CMAKE_CONFIGURATION_TYPES}")
  message("Default Build Configuration: ${_default_build_type}")
endif ()
message("Enabled PPMs:")
if (ROSETTA_PPM_SERIAL)
  message("  Serial:          ${CMAKE_CXX_COMPILER} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
  list(APPEND _disabled_ppm "Serial")
endif ()
if (ROSETTA_PPM_CUDA)
  message("  CUDA:            ${CMAKE_CUDA_COMPILER} ${_extra_flags} (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION})")
else ()
list(APPEND _disabled_ppm "CUDA")
endif ()
if (ROSETTA_PPM_OPENCL)
  message("  OpenCL:          ${CMAKE_CXX_COMPILER} ${OpenCL_LIBRARIES} ${_extra_flags} (${OpenCL_VERSION_STRING})")
else ()
list(APPEND _disabled_ppm "OpenCL")
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  message("  OpenMP-Parallel: ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
list(APPEND _disabled_ppm "OpenMP-Parallel")
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  message("  OpenMP-Task:     ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
list(APPEND _disabled_ppm "OpenMP-Task")
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  message("  OpenMP-Target:   ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
list(APPEND _disabled_ppm "OpenMP-Target")
endif ()
if (ROSETTA_PPM_OPENACC)
  message("  OpenACC:         ${CMAKE_CXX_COMPILER} ${OpenACC_CXX_FLAGS} ${OPENMP_OFFLOADING_CFLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${OpenACC_CXX_VERSION} ${OpenACC_CXX_SPEC_DATE})")
else ()
list(APPEND _disabled_ppm "OpenACC")
endif ()
list(JOIN _disabled_ppm " " _disabled_ppm_str)
message("Disabled PPMs: ${_disabled_ppm_str}")
message("Enabled Platforms:")
if (ROSETTA_PLATFORM_NVIDIA)
  message("  NVIDIA")
else ()
  list(APPEND _disabled_platform "NVIDIA")
endif ()
list(JOIN _disabled_platform " " _disabled_platform_str)
message("Disabled Platforms: ${_disabled_platform_str}")
message("")