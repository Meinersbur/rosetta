# Make the Rosetta Python module available for gen-stage1/gen-stage2/synth generators
cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST _pythonpath)
list(APPEND _pythonpath "${CMAKE_SOURCE_DIR}/rosetta/src")
cmake_path(CONVERT "${_pythonpath}" TO_NATIVE_PATH_LIST _pythonpath)
set(ENV{PYTHONPATH} "${_pythonpath}")

# Stage 1 #######################################################################################################

if (NOT ROSETTA_MAINTAINER_MODE)
  # These are the files gen-stage1.py looks for. If a new file is added, need to re-run CMake to search for build instructions.
  file(GLOB_RECURSE _potential_build_files FOLLOW_SYMLINKS LIST_DIRECTORIES false CONFIGURE_DEPENDS "*.cxx" "*.cu" "*.build")
endif ()

# internal TODO: basename is not required
function (add_benchmark_ppm basename ppm ppm_shortname _library)
  cmake_parse_arguments(_arg "" "PBSIZE" "SOURCES" ${ARGN})

  message("Add benchmark ${basename} ${ppm}")

  file(RELATIVE_PATH _relpath "${CMAKE_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}")
  string(REPLACE "/" "." _relpath_dots "${_relpath}")
  set(benchname "${basename}")
  set(_benchprop_filename "${CMAKE_CURRENT_BINARY_DIR}/${benchname}.${ppm_shortname}.$<CONFIG>.benchprop.cxx")

  set(_bench_target "${basename}.${ppm_shortname}")
  set(_sources ${_arg_SOURCES} "${_benchprop_filename}")

  string(REPLACE "-" "_" ppm_mixed "${ppm}")

  add_executable("${_bench_target}" ${_sources})
  set_target_properties(
    "${_bench_target}"
    PROPERTIES FOLDER "Benchmarks/${ppm}"
               ROSETTA_PPM "${ppm}"
               ROSETTA_BASENAME "${basename}"
               ROSETTA_BENCHPROPFILE "${_benchprop_filename}"
               ROSETTA_PBSIZE "${_arg_PBSIZE}")
#set_target_properties("${_bench_target}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON) #TODO: Remove if not needed
  target_link_libraries("${_bench_target}" PRIVATE "${_library}")
  if (ROSETTA_REALTYPE)
    # TODO: Support compiling mutiple real types at the same time?
    target_compile_definitions(${_bench_target} PRIVATE "ROSETTA_REALTYPE=${ROSETTA_REALTYPE}")
  endif ()

  add_dependencies("build-${ppm}" "${_bench_target}")

  append_global_property(benchmarks "${_bench_target}")
endfunction ()

function (add_benchmark_serial basename)
  if (NOT ROSETTA_PPM_SERIAL)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "serial" "serial" "rosetta-serial" ${ARGN})
endfunction ()

function (add_benchmark_cuda basename)
  if (NOT ROSETTA_PPM_CUDA)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "cuda" "cuda" "rosetta-cuda" ${ARGN})
endfunction ()

function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-parallel" "omp_parallel" "rosetta-openmp-parallel" ${ARGN})
endfunction ()

function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-task" "omp_task" "rosetta-openmp-task" ${ARGN})
endfunction ()

function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-target" "omp_target" "rosetta-openmp-target" ${ARGN})
endfunction ()

function (add_benchmark_sycl basename)
  if (NOT ROSETTA_PPM_SYCL)
    return()
  endif ()
  add_benchmark_ppm("${basename}" "sycl" "sycl" "rosetta-sycl" ${ARGN})
endfunction ()

# https://stackoverflow.com/a/52714922
set(_refsizeinclude "${CMAKE_CURRENT_BINARY_DIR}/stage1.cmake")
set(_stage1generator "${CMAKE_SOURCE_DIR}/rosetta/src/rosetta/scripts/gen-stage1.py")

# Python::Interpreter does not work?
set(_stage1args "--builddir=${CMAKE_CURRENT_BINARY_DIR}" "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}"
                "--problemsizefile=${_problemsizefile}" "--output=${_refsizeinclude}")
if (ROSETTA_CONFIGNAME)
  list(APPEND "--configname=${ROSETTA_CONFIGNAME}")
endif ()
separate_arguments(_filter_args NATIVE_COMMAND "${ROSETTA_BENCH_FILTER}")
list(APPEND _stage1args ${_filter_args})
message("Executing: ${Python_EXECUTABLE} -m rosetta.scripts.gen-stage1 ${_stage1args}")
execute_process(COMMAND "${Python_EXECUTABLE}" -m rosetta.scripts.gen-stage1 ${_stage1args} COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_CONFIGURE_DEPENDS ${_stage1generator})

# execute_process( COMMAND "${Python_EXECUTABLE}" "${_stage1generator}" "--builddir=${CMAKE_CURRENT_BINARY_DIR}"
# "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}" "--problemsizefile=${_problemsizefile}" "--output=${_refsizeinclude}"
# "--configname=${ROSETTA_CONFIGNAME}'"
# ${ROSETTA_BENCH_FILTER} COMMAND_ERROR_IS_FATAL ANY )

# Mark the input files as configure step dependencies so that the execute_process commands are retriggered on input file
# change. file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/generated/cmake/input_files _input_files) TODO: add_property
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${_problemsizefile};${_stage1generator}")

# Add the generated CMake targets.
include("${_refsizeinclude}")

get_property(_benchmarks GLOBAL PROPERTY benchmarks)

# Collect all config-specific compiler flags as generator expression
set(_config_cxx_flags)
foreach (_conftype IN LISTS _build_types)
  string(TOUPPER "${_conftype}" _conftype_upper)
  set(_config_cxx_flags "${_config_cxx_flags}$<$<CONFIG:${_conftype}>:${CMAKE_CXX_FLAGS_${_conftype_upper}}>")
endforeach ()

set(_benchmarks_run)
foreach (_target IN LISTS _benchmarks)
  get_property(
    _ppm
    TARGET "${_target}"
    PROPERTY ROSETTA_PPM)
  get_property(
    _basename
    TARGET "${_target}"
    PROPERTY ROSETTA_BASENAME)
  get_property(
    _benchpropfile
    TARGET "${_target}"
    PROPERTY ROSETTA_BENCHPROPFILE)
  get_property(
    _pbsize
    TARGET "${_target}"
    PROPERTY ROSETTA_PBSIZE)

  list(
    APPEND
    _benchmarks_run
    "register_benchmark(basename='${_basename}', ppm='${_ppm}', target='${_target}', exepath='$<TARGET_FILE:${_target}>', buildtype='$<CONFIG>', configname='${ROSETTA_CONFIGNAME}', benchpropfile='${_benchpropfile}', compiler='${CMAKE_CXX_COMPILER}', compilerflags='${CMAKE_CXX_FLAGS} ${_config_cxx_flags} $<TARGET_PROPERTY:${_target},COMPILE_OPTIONS> $<TARGET_PROPERTY:${_target},COMPILE_FLAGS>', pbsize=${_pbsize})"
  )
endforeach ()
string(JOIN "\n" _benchmarks_run ${_benchmarks_run})

# TODO: Which configuration we want depends on the PPM. E.g for CUDA, $<COMPILE_LANGUAGE:CUDA> shouold be used for
# compiler flags
file(
  GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchlist-$<CONFIG>$<IF:$<COMPILE_LANGUAGE:CXX>,,-$<COMPILE_LANGUAGE>>.py"
  CONTENT
    "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
  import sys
  sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')
from rosetta.registry import register_benchmark

${_benchmarks_run}

if __name__ == '__main__':
  runner_main()
"
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

if (_default_build_type)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "benchlist-${_default_build_type}.py" "benchlist.py"
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif ()


# Stage 2 #######################################################################################################

set(_benchpropfiles)
foreach (_benchmark IN LISTS _benchmarks)
  get_property(
    _benchpropfile
    TARGET "${_benchmark}"
    PROPERTY ROSETTA_BENCHPROPFILE)
  list(APPEND _benchpropfiles "${_benchpropfile}")
endforeach ()


set(_stage2generator "${CMAKE_SOURCE_DIR}/rosetta/src/rosetta/scripts/gen-stage2.py")
set(_stage2args
    "--builddir=${CMAKE_CURRENT_BINARY_DIR}" "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}"
    "--resultsdir=${ROSETTA_RESULTS_DIR}" "--benchfile=${CMAKE_CURRENT_BINARY_DIR}/benchlist-$<CONFIG>.py"
    "--config=$<CONFIG>")
add_custom_command(
  OUTPUT ${_benchpropfiles}
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/benchlist-$<CONFIG>.py" "${_stage2generator}"
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_pythonpath}" "$<TARGET_FILE:Python::Interpreter>" -m
          rosetta.scripts.gen-stage2 ${_stage2args}
  COMMENT "Generate properties files")

add_custom_target(genfiles DEPENDS ${_benchpropfiles})
