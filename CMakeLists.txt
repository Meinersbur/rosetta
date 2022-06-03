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


# TODO: use set_property(GLOBAL APPEND benchmarks_serial "${_target}")
function (append_global_property propname)
  get_property(propval GLOBAL PROPERTY ${propname})
  list(APPEND propval ${ARGN})
  set_property(GLOBAL PROPERTY ${propname} "${propval}")
endfunction ()


if (CMAKE_CONFIGURATION_TYPES)
  set(_build_types ${CMAKE_CONFIGURATION_TYPES})
elseif (CMAKE_BUILD_TYPE)
  set(_build_types "${CMAKE_BUILD_TYPE}")
else ()
  message(FATAL_ERROR "Cannot determine the build types")
endif ()


if (CMAKE_BUILD_TYPE)
  set(_default_build_type "${CMAKE_BUILD_TYPE}")
elseif (CMAKE_DEFAULT_BUILD_TYPE) # Only Ninja Multi-Config Generator
  set(_default_build_type "${CMAKE_DEFAULT_BUILD_TYPE}")
elseif ("Release" IN_LIST _build_types)
  set(_default_build_type "Release")
  # By default, Debug is first in CMAKE_CONFIGURATION_TYPES and becomes the implicit default. But we want Release to be the default by default.
  # CMake throws an error if setting CMAKE_DEFAULT_BUILD_TYPE and the generator does not support it
  if (CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
    set(CMAKE_DEFAULT_BUILD_TYPE "Release")
  endif ()
elseif (CMAKE_CONFIGURATION_TYPES)
  list(GET CMAKE_CONFIGURATION_TYPES 0 _default_build_type)
else ()
  set(_default_build_type)
endif ()



# TODO: should call something in bin/
set(RUNNER_EXECUTABLE "${CMAKE_SOURCE_DIR}/rosetta/lib/runner.py")



## Dependencies ################################################################

#set(CMAKE_CXX_STANDARD 17)

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


# For Nvidia platform
if (NOT DEFINED ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit)
else (ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit REQUIRED)
endif ()


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

set(_nvidia_enabled_default OFF)
if (TARGET CUDA::cudart)
  set(_nvidia_enabled_default ON)
endif ()
option(ROSETTA_PLATFORM_NVIDIA "Enable NVIDIA CUDA Toolkit/SDK as a platorm" ${_nvidia_enabled_default})

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

add_custom_target(build)
# TODO: build-bench
add_custom_target(build-verify)

if (ROSETTA_PPM_SERIAL)
  add_custom_target(build-serial)
  add_custom_target(build-serial-verify)
  add_dependencies(build build-serial build-serial-verify)
  add_dependencies(build-verify build-serial-verify)
endif ()
if (ROSETTA_PPM_CUDA)
  add_custom_target(build-cuda)
  add_custom_target(build-cuda-verify)
  add_dependencies(build build-cuda)
  add_dependencies(build-verify build-cuda-verify)
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK OR ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp)
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  add_custom_target(build-openmp-parallel)
  add_custom_target(build-openmp-parallel-verify)
  add_dependencies(build build-openmp-parallel)
  add_dependencies(build-verify build-openmp-parallel-verify)
  add_dependencies(build-openmp build-openmp-parallel)
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  add_custom_target(build-openmp-task)
  add_custom_target(build-openmp-task-verify)
  add_dependencies(build build-openmp-task)
  add_dependencies(build-verify build-openmp-task-verify)
  add_dependencies(build-openmp build-openmp-task)
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp-target)
  add_custom_target(build-openmp-target-verify)
  add_dependencies(build build-openmp-target)
  add_dependencies(build-verify build-openmp-target-verify)
  add_dependencies(build-openmp build-openmp-target)
endif ()


add_subdirectory(benchmarks)



## Run #########################################################################

get_property(_benchmarks_serial GLOBAL PROPERTY benchmarks_serial)
get_property(_verifications_serial GLOBAL PROPERTY verifications_serial)
get_property(_references GLOBAL PROPERTY references)


# TODO: separate verifications lists should not be necessary
list(APPEND _benchmarks ${_benchmarks_serial})
get_property(_benchmarks_cuda GLOBAL PROPERTY benchmarks_cuda)
get_property(_verifications_cuda GLOBAL PROPERTY verifications_cuda)
list(APPEND _benchmarks ${_benchmarks_cuda})
get_property(_benchmarks_openmp_parallel GLOBAL PROPERTY benchmarks_openmp_parallel)
get_property(_verifications_openmp_parallel GLOBAL PROPERTY verifications_openmp_parallel)
list(APPEND _benchmarks ${_benchmarks_openmp_parallel})
get_property(_benchmarks_openmp_task GLOBAL PROPERTY benchmarks_openmp_task)
get_property(_verifications_openmp_task GLOBAL PROPERTY verifications_openmp_task)
list(APPEND _benchmarks ${_benchmarks_openmp_task})
get_property(_benchmarks_openmp_target GLOBAL PROPERTY benchmarks_openmp_target)
get_property(_verifications_openmp_target GLOBAL PROPERTY verifications_openmp_target)
list(APPEND _benchmarks ${_benchmarks_openmp_target})

set(_benchmarks_run_arglist)


function (enable_ppm name shortname displayname)
  cmake_parse_arguments(_arg "" "" "BENCHMARKS;VERIFICATIONS"  ${ARGN})

  append_global_property(enabled_ppms "${name}")

  set(_benchmarks_run)
  foreach (_target IN LISTS _arg_BENCHMARKS)
    list(APPEND _benchmarks_run "register_benchmark('${_target}', exepath='$<TARGET_FILE:${_target}>', config='$<CONFIG>', ppm='${shortname}')")
  endforeach ()
  foreach (_target IN LISTS _arg_VERIFICATIONS)
    get_property(_reffile TARGET ${_target} PROPERTY REFERENCE_FILE)
    list(APPEND _benchmarks_run "register_verification('${_target}', exepath='$<TARGET_FILE:${_target}>', refpath='${_reffile}', config='$<CONFIG>', ppm='${shortname}')")
  endforeach ()
  list(JOIN _benchmarks_run "\n" _benchmarks_pylist)

  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/run-${name}-$<CONFIG>.py"
    CONTENT "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
  import sys
  sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')
from rosetta import runner_main, register_benchmark, register_verification

${_benchmarks_pylist}

if __name__ == '__main__':
  runner_main()
"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  if (_default_build_type)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "run-${name}-${_default_build_type}.py" "run-${name}.py"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_custom_target("run-${name}"
      COMMAND Python::Interpreter "run-${name}.py"
      COMMENT "Run ${displayname} benchmarks ..."
      USES_TERMINAL VERBATIM
    )
    add_custom_target("verify-${name}"
      COMMAND Python::Interpreter "run-${name}.py" "--verify"
      COMMENT "Run ${displayname} correctness checks ..."
      USES_TERMINAL VERBATIM
    )
    add_dependencies("verify-${name}" "build-${name}")
  endif ()
endfunction()



if (ROSETTA_PPM_SERIAL)
  enable_ppm(serial serial Serial BENCHMARKS ${_benchmarks_serial} VERIFICATIONS ${_verifications_serial})
endif ()

if (ROSETTA_PPM_CUDA)
  enable_ppm(cuda cuda CUDA BENCHMARKS ${_benchmarks_cuda} VERIFICATIONS ${_verifications_cuda})
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL)
  enable_ppm(openmp-parallel omp_parallel "OpenMP parallel" BENCHMARKS ${_benchmarks_openmp_parallel} VERIFICATIONS ${_verifications_openmp_parallel}) 
endif ()

if (ROSETTA_PPM_OPENMP_TASK)
  enable_ppm(openmp-task omp_task "OpenMP Tasks" BENCHMARKS ${_benchmarks_openmp_task} VERIFICATIONS ${_verifications_openmp_task})
endif ()

if (ROSETTA_PPM_OPENMP_TARGET)
  enable_ppm(openmp-target omp_target "OpenMP Target Offloading" BENCHMARKS ${_benchmarks_openmp_target} VERIFICATIONS ${_verifications_openmp_target})
endif ()



get_property(enabled_ppms GLOBAL PROPERTY enabled_ppms)

set(_ppms_pylist)
foreach (_name IN LISTS enabled_ppms)
  list(APPEND _ppms_pylist "load_register_file('run-${_name}-$<CONFIG>.py')")
endforeach ()
list(JOIN _ppms_pylist "\n" _ppms_pylist)

list(JOIN _benchmarks_run_arglist ",\n  " _benchmarks_run_pyarglist)
file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/run-$<CONFIG>.py"
    CONTENT "#! /usr/bin/env python3
# -*- coding: utf-8 -*-
    
if __name__ == '__main__':
  import sys
  sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')

from rosetta import runner_main, load_register_file
${_ppms_pylist}

if __name__ == '__main__':
  runner_main()
"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )



foreach (_conftype IN LISTS _build_types)
  add_custom_target(run-${_conftype}
    DEPENDS ${_benchmarks}
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
    COMMAND Python::Interpreter "run.py" "--bench"
    COMMENT "Run all benchmarks ..."
    USES_TERMINAL VERBATIM
  )
  add_dependencies(run build)

  # TODO: Individial verify targets
  add_custom_target(verify
    COMMAND Python::Interpreter "run.py" "--verify"
    COMMENT "Verify all benchmarks ..."
    USES_TERMINAL VERBATIM
  )
  add_dependencies(verify build-verify ${_references})
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
  get_property(_cxx_standard TARGET rosetta-serial-bench PROPERTY "CXX_STANDARD")
  if (NOT _cxx_standard STREQUAL "")
    set (_cxx_standard "${CMAKE_CXX_STANDARD_DEFAULT}")
  endif ()
  message("  Serial:          ${CMAKE_CXX_COMPILER} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, C++${_cxx_standard})")
else ()
  list(APPEND _disabled_ppm "Serial")
endif ()
if (ROSETTA_PPM_CUDA)
  message("  CUDA:            ${CMAKE_CUDA_COMPILER} ${_extra_flags} (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}, CUDA ${CUDA_VERSION})")
else ()
list(APPEND _disabled_ppm "CUDA")
endif ()
if (ROSETTA_PPM_OPENCL)
  message("  OpenCL:          ${CMAKE_CXX_COMPILER} ${OpenCL_LIBRARIES} ${_extra_flags} (OpenCL ${OpenCL_VERSION_STRING})")
else ()
list(APPEND _disabled_ppm "OpenCL")
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  message("  OpenMP-Parallel: ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})")
else ()
list(APPEND _disabled_ppm "OpenMP-Parallel")
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  message("  OpenMP-Task:     ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})")
else ()
list(APPEND _disabled_ppm "OpenMP-Task")
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  message("  OpenMP-Target:   ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_CFLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})")
else ()
list(APPEND _disabled_ppm "OpenMP-Target")
endif ()
if (ROSETTA_PPM_OPENACC)
  message("  OpenACC:         ${CMAKE_CXX_COMPILER} ${OpenACC_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenACC ${OpenACC_CXX_VERSION} ${OpenACC_CXX_SPEC_DATE})")
else ()
list(APPEND _disabled_ppm "OpenACC")
endif ()
list(JOIN _disabled_ppm " " _disabled_ppm_str)
message("Disabled PPMs: ${_disabled_ppm_str}")
message("Enabled Platforms:")
if (ROSETTA_PLATFORM_NVIDIA)
  message("  NVIDIA (CUDA ${CUDAToolkit_VERSION})")
else ()
  list(APPEND _disabled_platform "NVIDIA")
endif ()
list(JOIN _disabled_platform " " _disabled_platform_str)
message("Disabled Platforms: ${_disabled_platform_str}")
message("")
