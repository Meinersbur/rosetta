cmake_minimum_required(VERSION 3.19) # COMMAND_ERROR_IS_FATAL
project(
  rosetta
  VERSION 0.0.2
  LANGUAGES C CXX)

include(CheckLanguage)
include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)

#Required for IntelSYCL :added by AK
if (CMAKE_HOST_WIN32)
    # need at least CMake 3.23 for IntelLLVM support of IntelDPCPP package on Windows
    cmake_minimum_required(VERSION 3.23)
else()
    # CMake 3.20.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.20.5)
endif()


set(CMAKE_CTEST_ARGUMENTS "-j")
set(CTEST_TEST_TARGET_ALIAS check)
include(CTest)
set_target_properties(check PROPERTIES FOLDER "Self-Testing")

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/rosetta/cmake/modules")
include(Formatting)

# Use Folders in Visual Studio IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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
  message(FATAL_ERROR "Cannot determine the build type(s). Try -DCMAKE_BUILD_TYPE=Release")
endif ()

if (CMAKE_BUILD_TYPE)
  set(_default_build_type "${CMAKE_BUILD_TYPE}")
elseif (CMAKE_DEFAULT_BUILD_TYPE) # Only Ninja Multi-Config Generator
  set(_default_build_type "${CMAKE_DEFAULT_BUILD_TYPE}")
elseif ("Release" IN_LIST _build_types)
  set(_default_build_type "Release")
  # By default, Debug is first in CMAKE_CONFIGURATION_TYPES and becomes the implicit default. But we want Release to be
  # the default by default. CMake throws an error if setting CMAKE_DEFAULT_BUILD_TYPE and the generator does not support
  # it
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
set(GENCONFIG_EXECUTABLE "${CMAKE_SOURCE_DIR}/rosetta/bin/genconfig.py")

# TODO: Better name; "maintainer mode" comes from GNU autotools
option(
  ROSETTA_MAINTAINER_MODE
  "Skip always looking for changed benchmark build instructions (maintainer mode meaning now its the benchmark maintainer's responsibility to invoke CMake when benchmark build instructions change)"
  OFF)

# Dependencies ################################################################

# Python
find_package(Python 3.6 REQUIRED)

# Google Benchmark set(BENCHMARK_ENABLE_TESTING OFF) include(FetchContent) FetchContent_Declare(googlebenchmark
# GIT_REPOSITORY https://github.com/google/benchmark.git ) FetchContent_MakeAvailable(googlebenchmark)

# Catch2
# Literally an example from https://cmake.org/cmake/help/latest/module/FetchContent.html
# Or: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
# TODO: Option to skip; ROSETTA_MAINTAINER_MODE(?)
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.3.2
  #FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Catch2)
#find_package(Catch2)
if (TARGET Catch2)
  set_target_properties(Catch2 PROPERTIES FOLDER "Self-Testing")
endif ()
if (TARGET Catch2WithMain)
  set_target_properties(Catch2WithMain PROPERTIES FOLDER "Self-Testing")
endif ()

# CUDA
if (NOT DEFINED ROSETTA_PPM_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES OFF) # ???
  check_language(CUDA)
  find_package(CUDAToolkit)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  endif ()
elseif (ROSETTA_PPM_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES OFF) # ???
  check_language(CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA enabled but no CUDA not found")
  endif ()
endif ()

# Add -G option to debug flags by default because CMake's default doesn't
if (CMAKE_CUDA_COMPILER)
  if (NOT ROSETTA_CUDA_IS_FIRST_CONFIG)
    string(STRIP "${CMAKE_CUDA_FLAGS_DEBUG}" CMAKE_CUDA_FLAGS_DEBUG)
    string(STRIP "${CMAKE_CUDA_FLAGS_DEBUG_INIT}" CMAKE_CUDA_FLAGS_DEBUG_INIT)
    message("Is first config '${CMAKE_CUDA_FLAGS_DEBUG}' == '${CMAKE_CUDA_FLAGS_DEBUG_INIT}'")
    if (DEFINED CMAKE_CUDA_FLAGS_DEBUG AND CMAKE_CUDA_FLAGS_DEBUG STREQUAL CMAKE_CUDA_FLAGS_DEBUG_INIT)
      message("Appending -G to CMAKE_CUDA_FLAGS_DEBUG")
      set(CMAKE_CUDA_FLAGS_DEBUG
          "${CMAKE_CUDA_FLAGS_DEBUG} -G"
          CACHE STRING "Flags used by the CUDA compiler during DEBUG builds." FORCE)
    endif ()
    string(STRIP "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}" CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
    string(STRIP "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT}" CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT)
    if (DEFINED CMAKE_CUDA_FLAGS_RELWITHDEBINFO AND CMAKE_CUDA_FLAGS_RELWITHDEBINFO STREQUAL
                                                    CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT)
      message("Appending -G to CMAKE_CUDA_FLAGS_RELWITHDEBINFO")
      set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
          "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -G"
          CACHE STRING "Flags used by the CUDA compiler during RELWITHDEBINFO builds." FORCE)
    endif ()
  endif ()
  set(ROSETTA_CUDA_IS_FIRST_CONFIG
      1
      CACHE INTERNAL "Used to set override CMake's default flags" FORCE)
endif ()

if (CMAKE_CUDA_COMPILER)
  # only for CUDA_VERSION
  #find_package(CUDA REQUIRED)
  find_package(Thrust REQUIRED CONFIG)
  thrust_create_target(Thrust) # https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
endif ()

#message(FATAL_ERROR "Stop here")

# For Nvidia platform
if (NOT DEFINED ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit)
elseif (ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit REQUIRED)
endif ()

# OpenCL
if (NOT DEFINED ROSETTA_PPM_OPENCL)
  find_package(OpenCL)
elseif (ROSETTA_PPM_OPENCL)
  find_package(OpenCL REQUIRED)
endif ()

# pthreads find_package(Threads REQUIRED)

# OpenMP
if (NOT DEFINED ROSETTA_PPM_OPENMP_PARALLEL
    OR ROSETTA_PPM_OPENMP_PARALLEL
    OR NOT DEFINED ROSETTA_PPM_OPENMP_TASK
    OR ROSETTA_PPM_OPENMP_TASK
    OR NOT DEFINED ROSETTA_PPM_OPENMP_TARGET
    OR ROSETTA_PPM_OPENMP_TARGET)
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
  endif ()
endif ()

# OpenACC
if (NOT DEFINED ROSETTA_PPM_OPENACC)
  find_package(OpenACC)
elseif (ROSETTA_PPM_OPENACC)
  find_package(OpenACC REQUIRED)
endif ()

#SYCL   :added by AK
if (NOT DEFINED ROSETTA_PPM_SYCL)
  find_package(IntelSYCL)
elseif (ROSETTA_PPM_SYCL)
  find_package(IntelSYCL REQUIRED)
endif ()


# MPI
if (NOT DEFINED ROSETTA_PPM_MPI)
  find_package(MPI)
elseif (ROSETTA_PPM_MPI)
  find_package(MPI REQUIRED)
endif ()

# Introspection ################################################################

# pointer restrict
check_cxx_source_compiles(
  "
        int f(void *restrict x);
        int main(void) {return 0;}
    "
  HAVE_CXX_RESTRICT)

check_cxx_source_compiles(
  "
        int f(void *__restrict x);
        int main(void) {return 0;}
    "
  HAVE_CXX___RESTRICT)
if (HAVE_CXX_RESTRICT)
  set(KEYWORD_RESTRICT "restrict")
  elif(HAVE_CXX___RESTRICT)
  set(KEYWORD_RESTRICT "__restrict")
else ()
  set(KEYWORD_RESTRICT "")
endif ()

check_include_file_cxx(filesystem HAS_INCLUDE_FILESYSTEM)

# g++ >= 8: charconv with integral types
check_include_file_cxx(charconv HAS_INCLUDE_CHARCONV)

# g++ >= 11: charconv with float types (https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html)
check_cxx_source_compiles(
  "
#include <charconv>

template <typename T>
void test_from_chars() {
  char *p;
  T v;
  std::from_chars(p, p, &v);
}

template <typename T>
void test_to_chars() {
  char *p;
  T val;
  std::to_chars(p, p, val,std::chars_format::general);
}

void foo() {
  test_from_chars<float>();
  test_to_chars<float>();
  test_from_chars<double>();
  test_to_chars<double>();
}
"
  HAS_INCLUDE_CHARCONV_FLOAT)

# Configuration ###############################################################

set(ROSETTA_RESULTS_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/results"
    CACHE PATH "Default path for results (.xml files)")
set(ROSETTA_BENCH_FILTER
    ""
    CACHE STRING "Benchmark filter switches")

string(
  RANDOM
  LENGTH 8
  ALPHABET "0123456789" _randseed_default)
set(ROSETTA_RANDSEED
    "${_randseed_default}"
    CACHE STRING "Seed for compile-time randomness")

option(ROSETTA_PPM_REFERENCE "Enable reference code; use serial if there is no explicit reference version" ON)

option(ROSETTA_PPM_SERIAL "Enable serial" ON)

set(_cuda_enabled_default OFF)
if (CMAKE_CUDA_COMPILER)
  set(_cuda_enabled_default ON)
endif ()
option(ROSETTA_PPM_CUDA "Enable CUDA PPM" ${_cuda_enabled_default})

set(_nvidia_enabled_default OFF)
if (TARGET CUDA::cudart)
  set(_nvidia_enabled_default ON)
endif ()
option(ROSETTA_PLATFORM_NVIDIA "Enable NVIDIA CUPTI Performance Measurements" ${_nvidia_enabled_default})

set(_papi_enabled_default OFF)
if (False)
  set(_papi_enabled_default ON)
endif ()
option(ROSETTA_PLATFORM_PAPI "Enable Performance-API Measurements (Not implemented)" ${_papi_enabled_default})

set(_ompwtime_enabled_default OFF)
if (OPENMP_FOUND)
  set(_ompwtime_enabled_default ON)
endif ()
option(ROSETTA_PLATFORM_OPENMP "Enable omp_get_wtime Measurements (Not implemented)" ${_ompwtime_enabled_default})

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


#IntelSYCL  :added by AK
set(_sycl_enabled_default OFF)
if (IntelSYCL_FOUND)
  set(_sycl_enabled_default ON)
endif ()
option(ROSETTA_PPM_SYCL "Enable SYCL PPM" ${_sycl_enabled_default})

set(_mpi_enabled_default OFF)
if (MPI_CXX_FOUND)
  set(_mpi _enabled_default ON)
endif ()
option(ROSETTA_PPM_MPI "Enable MPI PPM" ${_mpi_enabled_default})

set(ROSETTA_REALTYPE
    "double"
    CACHE STRING "Which type to prefer for floating-point values ('float', 'double', or 'long double')")
set_property(CACHE ROSETTA_REALTYPE PROPERTY STRINGS "float" "double" "long double")

set(ROSETTA_PROBLEMSIZE
    "medium"
    CACHE STRING "Problemsize to use (predefined or path to ini-file)")

set(_problemsizefile)
get_filename_component(_rosetta_predefined "${ROSETTA_PROBLEMSIZE}.problemsize.ini" ABSOLUTE BASE_DIR
                       "${CMAKE_SOURCE_DIR}/benchmarks")
get_filename_component(_rosetta_filename "${ROSETTA_PROBLEMSIZE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}/benchmarks")
get_filename_component(_rosetta_fullpath "${ROSETTA_PROBLEMSIZE}" ABSOLUTE)
if (NOT _problemsizefile AND EXISTS "${_rosetta_predefined}")
  set(_problemsizefile "${_rosetta_predefined}")
endif ()
if (NOT _problemsizefile AND EXISTS "${_rosetta_filename}")
  set(_problemsizefile "${_rosetta_filename}")
endif ()
if (NOT _problemsizefile AND EXISTS "${_rosetta_fullpath}")
  set(_problemsizefile "${_rosetta_fullpath}")
endif ()
if (NOT _problemsizefile)
  message(SEND_ERROR "Problem sizes required")
endif ()

# Build #######################################################################

add_custom_target(references)
set_target_properties(references PROPERTIES FOLDER "Benchmarks")

set(_reference_dir "${PROJECT_BINARY_DIR}/references")
function (rosetta_add_reference basename pbsize)
  set(_reffile "${_reference_dir}/${basename}.${pbsize}.reference_output")
  set(_bench_target "${basename}.serial") # TODO: Allow non-serial

  add_custom_command(
    OUTPUT "${_reffile}"
    DEPENDS "${_bench_target}"
    COMMAND "${_bench_target}" --verify --verifyfile "${_reffile}"
    COMMENT "Generate ${basename} reference output (using serial output) at ${_reffile}...")
  add_custom_target("reference-${basename}" DEPENDS "${_reffile}")
  set_target_properties("reference-${basename}" PROPERTIES FOLDER "Reference outputs")
  add_dependencies(references "reference-${basename}")
endfunction ()

# Library ----------------------------------------------------------------------

add_subdirectory(rosetta)

# Benchmarks -------------------------------------------------------------------

add_custom_target(build)
if (ROSETTA_PPM_SERIAL)
  add_custom_target(build-serial)
  set_target_properties(build-serial PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build build-serial)
endif ()
if (ROSETTA_PPM_CUDA)
  add_custom_target(build-cuda)
  set_target_properties(build-cuda PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build build-cuda)

  # For convenience, we add -G to add debug symbols to device code by default, which CMake unfortunately forgets
  #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G" CACHE STRING "Flags used by the CUDA compiler during DEBUG builds.")
  #set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -G" CACHE STRING "Flags used by the CUDA compiler during RELWITHDEBINFO builds.")
  #message("CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL
    OR ROSETTA_PPM_OPENMP_TASK
    OR ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp)
  set_target_properties(build-openmp PROPERTIES FOLDER "Benchmarks")
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  add_custom_target(build-openmp-parallel)
  set_target_properties(build-openmp-parallel PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build-openmp build-openmp-parallel)
  add_dependencies(build build-openmp-parallel)
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  add_custom_target(build-openmp-task)
  set_target_properties(build-openmp-task PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build-openmp build-openmp-task)
  add_dependencies(build build-openmp-task)
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  add_custom_target(build-openmp-target)
  set_target_properties(build-openmp-target PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build-openmp build-openmp-target)
  add_dependencies(build build-openmp-target)
endif ()
#Added by AK
if (ROSETTA_PPM_SYCL)
  add_custom_target(build-sycl)
  set_target_properties(build-sycl PROPERTIES FOLDER "Benchmarks")
  add_dependencies(build build-sycl)
endif ()
add_subdirectory(benchmarks)

# Maintanance #################################################################

if (NOT ROSETTA_MAINTAINER_MODE)
  add_format_target(
    CLANG_RGLOB
    benchmarks/*.cxx
    benchmarks/*.cu
    benchmarks/*.h
    rosetta/*.h
    rosetta/*.cpp
    PY_RGLOB
    rosetta/src/*.py
    rosetta/tests/*.py
    CMAKE
    CMakeLists.txt
    CMAKE_RGLOB
    rosetta/CMakeLists.txt
    rosetta/cmake/*.cmake)
endif ()

# Run #########################################################################

get_property(_benchmarks GLOBAL PROPERTY benchmarks)

file(
  GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/bench-$<CONFIG>.py"
  CONTENT
    "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if not sys.version_info >= (3, 9):
    print('Requires python 3.9 or later', file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/src')

import pathlib
import rosetta.driver

thisscriptdir = pathlib.Path(__file__).parent
rosetta.driver.driver_main(
    mode=rosetta.driver.DriverMode.USERBUILDDIR,
    default_action=rosetta.driver.DefaultAction.BENCH,
    benchlistfile=thisscriptdir / 'benchmarks' / 'benchlist-$<CONFIG>.py',
    srcdir='${CMAKE_SOURCE_DIR}',
    builddir=thisscriptdir
  )
"
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
add_custom_target(
  bench
  COMMAND Python::Interpreter "bench-$<CONFIG>.py"
  COMMENT "Run benchmarks ..."
  USES_TERMINAL VERBATIM)
add_dependencies(bench ${_benchmarks})

file(
  GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/verify-$<CONFIG>.py"
  CONTENT
    "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if not sys.version_info >= (3, 9):
    print('Requires python 3.9 or later', file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/src')

import pathlib
import rosetta.driver

thisscriptdir = pathlib.Path(__file__).parent
rosetta.driver.driver_main(
    mode=rosetta.driver.DriverMode.USERBUILDDIR,
    default_action=rosetta.driver.DefaultAction.VERIFY,
    benchlistfile=thisscriptdir / 'benchmarks' / 'benchlist-$<CONFIG>.py',
    srcdir='${CMAKE_SOURCE_DIR}',
    builddir=thisscriptdir
  )
"
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
add_custom_target(
  verify
  COMMAND Python::Interpreter "verify-$<CONFIG>.py" --no-buildondemand
  COMMENT "Verify benchmarks ..."
  USES_TERMINAL VERBATIM)
add_dependencies(verify ${_benchmarks})

file(
  GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/probe-$<CONFIG>.py"
  CONTENT
    "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if not sys.version_info >= (3, 9):
    print('Requires python 3.9 or later', file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/src')

import pathlib
import rosetta.driver

thisscriptdir = pathlib.Path(__file__).parent
rosetta.driver.driver_main(
    mode=rosetta.driver.DriverMode.USERBUILDDIR,
    default_action=rosetta.driver.DefaultAction.PROBE,
    benchlistfile=thisscriptdir / 'benchmarks' / 'benchlist-$<CONFIG>.py',
    srcdir='${CMAKE_SOURCE_DIR}',
    builddir=thisscriptdir
  )
"
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
add_custom_target(
  probe
  COMMAND Python::Interpreter "probe-$<CONFIG>.py" --no-buildondemand
  COMMENT "Probe problemsize ..."
  USES_TERMINAL VERBATIM)
add_dependencies(probe ${_benchmarks})

if (_default_build_type)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "bench-${_default_build_type}.py" "bench.py"
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "verify-${_default_build_type}.py" "verify.py"
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  # add_custom_target(run COMMAND Python::Interpreter "run.py" "--problemsizefile=${_problemsizefile}" "--bench" COMMENT
  # "Run all benchmarks ..." USES_TERMINAL VERBATIM ) add_dependencies(run build)

  # add_custom_target(verify COMMAND Python::Interpreter "run.py" "--problemsizefile=${_problemsizefile}" "--verify"
  # COMMENT "Verify all benchmarks ..." USES_TERMINAL VERBATIM ) add_dependencies(verify build ${_references})
endif ()

# Summary #########################################################################

set(_disabled_ppm)
set(_disabled_platform)

set(_extra_flags)
if (ROSETTA_NO_EXCEPTIONS)
  set(_extra_flags "-fno-exceptions")
endif ()

get_filename_component(_problemsizefilename "${_problemsizefile}" NAME)

message("")
message("Configuration Summary")
message("---------------------")
if (CMAKE_BUILD_TYPE)
  message("Build Configuration: ${CMAKE_BUILD_TYPE}")
else ()
  message("Build Configurations: ${CMAKE_CONFIGURATION_TYPES}")
  message("Default Build Configuration: ${_default_build_type}")
endif ()
message("Default problem size: ${_problemsizefilename}")
message("Enabled PPMs:")
if (ROSETTA_PPM_SERIAL)
  get_property(
    _cxx_standard
    TARGET rosetta-serial
    PROPERTY CXX_STANDARD)
  if (_cxx_standard STREQUAL "")
    set(_cxx_standard "${CMAKE_CXX_STANDARD}")
  endif ()
  if (_cxx_standard STREQUAL "")
    set(_cxx_standard "${CMAKE_CXX_STANDARD_DEFAULT}")
  endif ()
  message(
    "  Serial:          ${CMAKE_CXX_COMPILER} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, C++${_cxx_standard})"
  )
else ()
  list(APPEND _disabled_ppm "Serial")
endif ()
if (ROSETTA_PPM_CUDA)
  message(
    "  CUDA:            ${CMAKE_CUDA_COMPILER} ${_extra_flags} (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}, CUDA ${CUDA_VERSION})"
  )
else ()
  list(APPEND _disabled_ppm "CUDA")
endif ()
if (ROSETTA_PPM_OPENCL)
  message(
    "  OpenCL:          ${CMAKE_CXX_COMPILER} ${OpenCL_LIBRARIES} ${_extra_flags} (OpenCL ${OpenCL_VERSION_STRING})")
else ()
  list(APPEND _disabled_ppm "OpenCL")
endif ()
if (ROSETTA_PPM_OPENMP_PARALLEL)
  message(
    "  OpenMP-Parallel: ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})"
  )
else ()
  list(APPEND _disabled_ppm "OpenMP-Parallel")
endif ()
if (ROSETTA_PPM_OPENMP_TASK)
  message(
    "  OpenMP-Task:     ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})"
  )
else ()
  list(APPEND _disabled_ppm "OpenMP-Task")
endif ()
if (ROSETTA_PPM_OPENMP_TARGET)
  message(
    "  OpenMP-Target:   ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_CFLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenMP ${OpenMP_CXX_VERSION} ${OpenMP_CXX_SPEC_DATE})"
  )
else ()
  list(APPEND _disabled_ppm "OpenMP-Target")
endif ()
if (ROSETTA_PPM_OPENACC)
  message(
    "  OpenACC:         ${CMAKE_CXX_COMPILER} ${OpenACC_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, OpenACC ${OpenACC_CXX_VERSION} ${OpenACC_CXX_SPEC_DATE})"
  )
else ()
  list(APPEND _disabled_ppm "OpenACC")
endif ()
if (ROSETTA_PPM_SYCL) #added by AK
  message(
    "  SYCL: ${SYCL_COMPILER} ${SYCL_CXX_FLAGS} ${_extra_flags} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, IntelSYCL ${SYCL_LANGUAGE_VERSION})"
  )
else ()
  list(APPEND _disabled_ppm "OpenMP-Parallel")
endif ()
if (ROSETTA_PPM_MPI)
  message(
    "  MPI:         ${MPI_CXX_COMPILER} ${MPI_CXX_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_DEFINITIONS} ${MPI_CXX_INLCUDE_DIRS} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} (${MPI_CXX_VERSION})"
  )
else ()
  list(APPEND _disabled_ppm "MPI")
endif ()
list(JOIN _disabled_ppm " " _disabled_ppm_str)
message("Disabled PPMs: ${_disabled_ppm_str}")
message("Enabled Platforms:")
if (ROSETTA_PLATFORM_NVIDIA)
  message("  NVIDIA (CUDAToolkit ${CUDAToolkit_VERSION})")
else ()
  list(APPEND _disabled_platform "NVIDIA")
endif ()
list(JOIN _disabled_platform " " _disabled_platform_str)
message("Disabled Platforms: ${_disabled_platform_str}")

message("")
message("Cheat Sheet (What can I do now?)")
message("--------------------------------")
# TODO: Customize to driver mode/local path
message("Build:                     ninja (implicit for every command)")
if (ROSETTA_MAINTAINER_MODE)
  message("Search for new benchmarks: cmake .")
endif ()
message("")
message("Probe max problemsize:     ./probe.py --limit-walltime=5s")
message("Performance tuning:        <TODO>")
message("Verify correctness:        ./verify.py [--problemsizefile PATH]")
message("Run benchmarks:            ./bench.py [--problemsizefile PATH]")
message("Analyze previous runs:     <TODO>")
