cmake_minimum_required(VERSION 3.19) # COMMAND_ERROR_IS_FATAL
project(
  rosetta
  VERSION 0.0.2
  LANGUAGES CXX)

include(CheckLanguage)
include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)

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

# TODO: Better name; "maintainer mode" comes from GNU autotools
option(
  ROSETTA_MAINTAINER_MODE
  "Skip always looking for changed benchmark build instructions (maintainer mode meaning now its the benchmark maintainer's responsibility to invoke CMake when benchmark build instructions change)"
  OFF)





set(ROSETTA_PPM_DEFAULT
  "AUTODETECT"
  CACHE STRING "Default enabled state of PPMs (ON, OFF, or AUTODETECT)")
set_property(CACHE ROSETTA_PPM_DEFAULT PROPERTY STRINGS "ON" "OFF" "AUTODETECT")

set(ROSETTA_PLATFORM_DEFAULT
  "AUTODETECT"
  CACHE STRING "Default enabled state of PLATFORMs (ON, OFF, or AUTODETECT)")
set_property(CACHE ROSETTA_PLATFORM_DEFAULT PROPERTY STRINGS "ON" "OFF" "AUTODETECT")



set(all_PPMS)
set(all_Ppms)
# The following variables are defined per-PPM (where * is the name of the PPM):
#
# - ROSETTA_PPM_*: Is the PPM Enabled? (cached, user-controlled except on first run)
#   - ON:        Compile benchmarks of this PPM and fail if dependencies are not met
#   - OFF:       Do not compile benchmarks of this PPM and do not search for its dependencies (unless required by another PPM/PLATFORM)
#   - undefined: Introspect the environment whether the PPM dependencies are met. 
#            If all dependencies found: Set the option default to ON, this will be the option's value during the next run of cmake unless overriden by the user
#            Otherwise:                 Set the option default to OFF, this will be the option's value during the next run of cmake unless overriden by the user
# - displayname_ppm_*: Name to be displayed to be the user (may contain hypens etc.)
# - helptext_ppm_*: Enabled option help string
# - autodetect_ppm_*: Autodetection result
macro (add_ppm _Ppm)
  cmake_parse_arguments(_arg "NOPREREQ" "DISPLAYNAME;ENABLEDHELP" "" ${ARGN})
  
  string(TOUPPER "${_Ppm}" _PPM)
  string(TOLOWER "${_Ppm}" _ppm)
  list(APPEND all_PPMS "${_PPM}")
  list(APPEND all_Ppms "${_Ppm}")

  set (_displayname "${_Ppm}")
  if (_arg_DISPLAYNAME)
    set(_displayname "${_arg_DISPLAYNAME}" )
  endif ()
  set (displayname_ppm_${_ppm} "${_displayname}") 

  if (_arg_NOPREREQ)
    set(autodetect_ppm_${_ppm} "ON")
  endif ()

  if (_arg_HELP_ENABLED)
    set (helptext_ppm_${_ppm} "${_arg_ENABLEDHELP}") 
  else ()
    set (helptext_ppm_${_ppm} "Enable ${_displayname} PPM") 
  endif () 

  # Set default option to skip autodetection
  # If first configuring with autodetect, the PPM variable will undefined and later trigger weak detection of dependencies
  if (NOT ROSETTA_PPM_DEFAULT STREQUAL "AUTODETECT" AND NOT DEFINED ROSETTA_PPM_${_PPM})
    set(ROSETTA_PPM_${_PPM} ${ROSETTA_PPM_DEFAULT})
  endif ()
endmacro ()



add_ppm(Serial NOPREREQ)
add_ppm(CUDA)
add_ppm(OpenCL)
add_ppm(OpenMP_Parallel DISPLAYNAME "OpenMP-Parallel"
  ENABLEDHELP "Enable OpenMP '#pragma omp parallel' PPM")
add_ppm(OpenMP_Task DISPLAYNAME "OpenMP-Task"
  ENABLEDHELP "Enable OpenMP '#pragma omp task' PPM")
add_ppm(OpenMP_Target DISPLAYNAME "OpenMP-Target"
  ENABLEDHELP "Enable OpenMP '#pragma omp target' PPM")
add_ppm(SYCL)
add_ppm(OpenACC)
add_ppm(MPI)




set(all_Platforms)
# The following variables are defined per-platform (where * is the name of the platform):
#
# - ROSETTA_PLATFORM_*: Is the platform Enabled? (cached, user-controlled except on first run)
# - displayname_ppm_*: Name to be displayed to be the user (may contain hypens etc.)
# - helptext_platform_*: Enabled option help string
# - autodetect_ppm_*: Autodetection result
macro (add_platform _Platform)
  cmake_parse_arguments(_arg "NOPREREQ" "DISPLAYNAME;ENABLEDHELP" "" ${ARGN})
  
  string(TOUPPER "${_Platform}" _PLATFORM)
  string(TOLOWER "${_Platform}" _platform)
  list(APPEND all_Platforms "${_Platform}")

  set (_displayname "${_Platform}")
  if (_arg_DISPLAYNAME)
    set(_displayname "${_arg_DISPLAYNAME}" )
  endif ()
  set ("displayname_platform_${_platform}" "${_displayname}") 

  if (_arg_NOPREREQ)
    set(autodetect_platform_${_platform} "ON")
  endif ()

  if (_arg_HELP_ENABLED)
    set (helptext_platform_${_platform} "${_arg_ENABLEDHELP") 
  else ()
    set (helptext_platform_${_platform} "Enable ${_displayname} Platform") 
  endif ()

  # Set default option to skip autodetection
  if (NOT ROSETTA_PLATFORM_DEFAULT STREQUAL "AUTODETECT" AND NOT DEFINED ROSETTA_PLATFORM_${_PLATFORM})
    set(ROSETTA_PLATOFRM_${_PPM} ${ROSETTA_PLATFORM_DEFAULT})
  endif ()
endmacro ()

add_platform(NVIDIA 
  ENABLEDHELP "Enable NVIDIA CUPTI Performance Measurements")
add_platform(OpenMP
  ENABLEDHELP "Enable omp_get_wtime measurements")
add_platform(PAPI 
  ENABLEDHELP "Enable PAPI measurements (not implemented)")
add_platform(OMPT
  ENABLEDHELP "Enable OMPT measurements (not implemented)")



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
set(CMAKE_CUDA_ARCHITECTURES OFF) # ???

if (ROSETTA_PPM_CUDA OR ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit REQUIRED)
elseif (NOT DEFINED ROSETTA_PPM_CUDA OR NOT DEFINED ROSETTA_PLATFORM_NVIDIA)
  find_package(CUDAToolkit)
endif ()

if (ROSETTA_PPM_CUDA)
  enable_language(CUDA)
  find_package(Thrust REQUIRED CONFIG)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA enabled but no CUDA found")
  endif ()
  if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA enabled but no CUDA Toolkit found")
  endif ()
elseif (NOT DEFINED ROSETTA_PPM_CUDA)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(Thrust CONFIG)
  endif ()
endif ()

if (Thrust_FOUND)
  thrust_create_target(Thrust) # https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
endif ()


# For Nvidia platform
if (CUDAToolkit_FOUND)
  set(autodetect_platform_nvidia "ON")
endif ()

# For CUDA PPM
if (CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER AND Thrust_FOUND)
  set(autodetect_ppm_cuda "ON")
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
  set(ROSETTA_CUDA_IS_FIRST_CONFIG 1 CACHE INTERNAL "Used to override CMake's default flags" FORCE)
endif ()





# OpenCL
if (NOT DEFINED ROSETTA_PPM_OPENCL)
  find_package(OpenCL)
elseif (ROSETTA_PPM_OPENCL)
  find_package(OpenCL REQUIRED)
endif ()
if (OpenCL_FOUND)
  set(autodetect_ppm_opencl "ON")
endif ()


# pthreads 
# find_package(Threads REQUIRED)

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
if (OpenMP_CXX_FOUND)
  set(autodetect_ppm_openmp_parallel "ON")
  set(autodetect_ppm_openmp_task "ON")
  set(autodetect_platform_openmp "ON")
endif ()
if (OpenMPOffload_FOUND)
  set(autodetect_ppm_openmp_target "ON")
endif ()


#SYCL
if (NOT DEFINED ROSETTA_PPM_SYCL)
  find_package(IntelSYCL CONFIG QUIET)
  if (NOT IntelSYCL_FOUND)
    message(STATUS "IntelSYCL NOT found. Finding IntelDPCPP")
    find_package(IntelDPCPP CONFIG QUIET)
  endif ()
elseif (ROSETTA_PPM_SYCL)
  find_package(IntelSYCL CONFIG)
  if(IntelSYCL_FOUND)
    message(STATUS "Found package IntelSYCL")
  else ()
    message(STATUS "Package IntelSYCL not found")
    find_package(IntelDPCPP CONFIG REQUIRED)
    message(STATUS "Found package IntelDPCPP")
  endif ()
endif ()
if (IntelSYCL_FOUND OR IntelDPCPP_FOUND)
  set(autodetect_ppm_sycl "ON")
endif ()

# OpenACC
if (NOT DEFINED ROSETTA_PPM_OPENACC)
  find_package(OpenACC)
elseif (ROSETTA_PPM_OPENACC)
  find_package(OpenACC REQUIRED)
endif ()
if (OpenACC_CXX_FOUND)
  set(autodetect_ppm_openacc "ON")
endif ()


# MPI
if (NOT DEFINED ROSETTA_PPM_MPI)
  find_package(MPI)
elseif (ROSETTA_PPM_MPI)
  find_package(MPI REQUIRED)
endif ()
if (MPI_CXX_FOUND)
  set(autodetect_ppm_mpi "ON")
endif ()


# Introspection ################################################################

# pointer restrict
check_cxx_source_compiles("
        int f(void *restrict x);
        int main(void) {return 0;}
    "
  HAVE_CXX_RESTRICT)

check_cxx_source_compiles("
        int f(void *__restrict x);
        int main(void) {return 0;}
    "
  HAVE_CXX___RESTRICT)
if (HAVE_CXX_RESTRICT)
  set(KEYWORD_RESTRICT "restrict")
elseif (HAVE_CXX___RESTRICT)
  set(KEYWORD_RESTRICT "__restrict")
else ()
  set(KEYWORD_RESTRICT "")
endif ()

check_include_file_cxx(filesystem HAS_INCLUDE_FILESYSTEM)

# g++ >= 8: charconv with integral types
check_include_file_cxx(charconv HAS_INCLUDE_CHARCONV)

# g++ >= 11: charconv with float types (https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html)
check_cxx_source_compiles("
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

set(ROSETTA_REALTYPE
    "double"
    CACHE STRING "Which type to prefer for floating-point values ('float', 'double', or 'long double')")
set_property(CACHE ROSETTA_REALTYPE PROPERTY STRINGS "float" "double" "long double")

set(ROSETTA_PROBLEMSIZE
    "medium"
    CACHE STRING "Problemsize to use (predefined or path to ini-file)")

# Add option for enabling each PPM
foreach(_PPM IN LISTS all_PPMS)
  string(TOLOWER "${_PPM}" _ppm)
  if (ROSETTA_PPM_DEFAULT STREQUAL "AUTODETECT")
    if (autodetect_ppm_${_ppm})
      set(_enabled_default ON)
    else ()
      set(_enabled_default OFF)
    endif () 
  else ()
    set(_enabled_default ${ROSETTA_PPM_DEFAULT})
  endif ()
  option(ROSETTA_PPM_${_PPM} "${helptext_ppm_${_ppm}}" ${_enabled_default})
endforeach()

foreach(_PLATFORM IN LISTS all_PLATFORMS)
  string(TOLOWER "${_PLATFORM}" _platform)
  if (ROSETTA_PLATFORM_DEFAULT STREQUAL "AUTODETECT")
    if (autodetect_platform_${_platform})
      set(_enabled_default ON)
    else ()
      set(_enabled_default OFF)
    endif () 
  else ()
    set(_enabled_default ${ROSETTA_PLATFORM_DEFAULT})
  endif ()
  option(ROSETTA_PLATFORM_${_PLATFORM} "${helptext_ppm_${_ppm}}" ${_enabled_default})
endforeach()



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
# Disable formatting because clang-format does not handle #pragma well.
#    benchmarks/*.cxx
#    benchmarks/*.cu
#    benchmarks/*.h
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
    "#! ${Python_EXECUTABLE}
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
    "#! ${Python_EXECUTABLE}
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
    "#! ${Python_EXECUTABLE}
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


get_filename_component(_problemsizefilename "${_problemsizefile}" NAME)


set(_extra_flags)
if (ROSETTA_NO_EXCEPTIONS)
  set(_extra_flags "-fno-exceptions")
endif ()


set(clargs_ppm_serial "${CMAKE_CXX_COMPILER}" ${_extra_flags}) 
set(verinfo_ppm_serial "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

set(clargs_ppm_cuda ${CMAKE_CUDA_COMPILER} ${_extra_flags}) 
set(verinfo_ppm_cuda "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}" "CUDA ${CUDAToolkit_VERSION}")

set(clargs_ppm_opencl "${CMAKE_CXX_COMPILER}" ${OpenCL_LIBRARIES} ${_extra_flags}) 
set(verinfo_ppm_opencl "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "OpenCL ${OpenCL_VERSION_STRING}")

set(clargs_ppm_openmp_parallel ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags}) 
set(verinfo_ppm_openmp_parallel "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "OpenMP ${OpenMP_CXX_VERSION}")

set(clargs_ppm_openmp_task ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${_extra_flags}) 
set(verinfo_ppm_openmp_task "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "OpenMP ${OpenMP_CXX_VERSION}")

set(clargs_ppm_openmp_target ${CMAKE_CXX_COMPILER} ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_LDFLAGS} ${_extra_flags}) 
set(verinfo_ppm_openmp_target "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "OpenMP ${OpenMP_CXX_VERSION}")

set(clargs_ppm_sycl ${SYCL_COMPILER} ${SYCL_CXX_FLAGS} ${_extra_flags}) 
set(verinfo_ppm_sycl "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "IntelSYCL ${SYCL_LANGUAGE_VERSION}")

set(clargs_ppm_openacc ${CMAKE_CXX_COMPILER} ${OpenACC_CXX_FLAGS} ${_extra_flags}) 
set(verinfo_ppm_openacc "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "OpenACC ${OpenACC_CXX_VERSION}")

set(clargs_ppm_mpi ${MPI_CXX_COMPILER} ${MPI_CXX_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_DEFINITIONS} ${MPI_CXX_INCLUDE_DIRS} ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${_extra_flags}) 
set(verinfo_ppm_mpi "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "MPI ${MPI_CXX_VERSION}")

set(verinfo_platform_nvidia "CUDA ${CUDAToolkit_VERSION}")







function (add_padding _varname _targetlen)
  string(LENGTH "${${_varname}}" _strlen)
  math(EXPR _padlen "(${_targetlen}) - (${_strlen})")
  if (_padlen GREATER 0)
    string(REPEAT " " "${_padlen}" _pad)
    set("${_varname}" "${${_varname}}${_pad}" PARENT_SCOPE)
  endif ()
endfunction()


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
set(_disabled_ppm)
foreach (_Ppm IN LISTS all_Ppms)
  string(TOUPPER "${_Ppm}" _PPM)
  string(TOLOWER "${_Ppm}" _ppm)
  if (ROSETTA_PPM_${_PPM})
    set(_infostr "${displayname_ppm_${_ppm}}:")
    if (verinfo_platform_${_platform})
      string(JOIN ", " _joinedverinfo ${verinfo_platform_${_platform}})
      set(_infostr "${_infostr} (${_joinedverinfo})")
    endif ()


    if (TARGET rosetta-${_ppm})
        # TODO: For CUDA, this is "CUDA_STANDARD"
        get_property(
          _cxx_standard
          TARGET rosetta-${_ppm}
          PROPERTY CXX_STANDARD)
        if (_cxx_standard STREQUAL "")
          set(_cxx_standard "${CMAKE_CXX_STANDARD}")
        endif ()
        if (_cxx_standard STREQUAL "")
          set(_cxx_standard "${CMAKE_CXX_STANDARD_DEFAULT}")
        endif ()
        if (_cxx_standard)
          list (APPEND verinfo_ppm_${_ppm} "C++${_cxx_standard}")
        endif ()
    endif ()
    add_padding(_infostr 16)
    string(JOIN " " _joinedargs ${clargs_ppm_${_ppm}})
    set(_infostr "${_infostr} ${_joinedargs}")

    if (verinfo_ppm_${_ppm})
      string(JOIN ", " _joinedverinfo ${verinfo_ppm_${_ppm}})
      set(_infostr "${_infostr} (${_joinedverinfo})")
    endif ()

    message("  ${_infostr}")
  else ()
    list(APPEND _disabled_ppm "${displayname_ppm_${_ppm}}")
  endif ()
endforeach ()

list(JOIN _disabled_ppm " " _disabled_ppm_str)
message("Disabled PPMs: ${_disabled_ppm_str}")  

message("Enabled Platforms:")
set(_disabled_platform)
foreach (_Platform IN LISTS all_Platforms)
  string(TOUPPER "${_Platform}" _PLATFORM)
  string(TOLOWER "${_Platform}" _platform)
  if (ROSETTA_PLATFORM_${_PLATFORM})
    set(_infostr "${displayname_platform_${_platform}}")
    if (verinfo_platform_${_platform})
      string(JOIN ", " _joinedverinfo ${verinfo_platform_${_platform}})
      set(_infostr "${_infostr} (${_joinedverinfo})")
    endif ()
    message("  ${_infostr}")
  else ()
    list(APPEND _disabled_platform "${displayname_platform_${_platform}}")
  endif ()
endforeach ()
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
