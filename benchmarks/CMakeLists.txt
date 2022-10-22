




## Stage 1 #######################################################################################################

# internal
# TODO: basename is not required
function (add_benchmark_ppm basename ppm ppm_shortname _library)
  cmake_parse_arguments(_arg "" "PBSIZE" "SOURCES" ${ARGN})
  message("_arg_SOURCES=${_arg_SOURCES}")
  message("_arg_PBSIZE=${_arg_PBSIZE}")

  file(RELATIVE_PATH _relpath  "${CMAKE_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}")
  string(REPLACE "/" "." _relpath_dots "${_relpath}")
  set(benchname "${basename}")
  set(_benchprop_filename "${CMAKE_CURRENT_BINARY_DIR}/${benchname}.${ppm_shortname}.$<CONFIG>.benchprop.cxx")

  set(_bench_target "${basename}.${ppm_shortname}")
  set(_sources ${_arg_SOURCES} "${_benchprop_filename}")

  string(REPLACE "-" "_" ppm_mixed "${ppm}")

  add_executable("${_bench_target}" ${_sources})
  message("added target: ${_bench_target}")
  message("SOURCES: ${_sources}")
  set_target_properties("${_bench_target}" 
    PROPERTIES
      FOLDER "Benchmarks/${ppm}"
      ROSETTA_PPM "${ppm}"
      ROSETTA_BASENAME "${basename}"
      ROSETTA_BENCHPROPFILE "${_benchprop_filename}"
      ROSETTA_PBSIZE "${_arg_PBSIZE}"
    )
  target_link_libraries("${_bench_target}" PRIVATE "${_library}")
  if (ROSETTA_REALTYPE)
    # TODO: Support compiling mutiple real types at the same time?
    target_compile_definitions(${_bench_target} PRIVATE "ROSETTA_REALTYPE=${ROSETTA_REALTYPE}")
  endif ()

  add_dependencies("build-${ppm}" "${_bench_target}")

  append_global_property(benchmarks "${_bench_target}")
endfunction ()






function (add_benchmark_serial basename)
  message("add_benchmark_serial(${basename})")
  if (NOT ROSETTA_PPM_SERIAL)
    return ()
  endif ()
  add_benchmark_ppm("${basename}" "serial" "serial" "rosetta-serial" ${ARGN})
endfunction ()


function (add_benchmark_cuda basename)
  if (NOT ROSETTA_PPM_CUDA)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "cuda" "cuda" "rosetta-cuda" ${ARGN} )
endfunction ()


function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-parallel" "omp_parallel" "rosetta-openmp-parallel" ${ARGN} )
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-task" "omp_task" "rosetta-openmp-task" ${ARGN} )
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-target" "omp_target" "rosetta-openmp-target" ${ARGN} )
endfunction ()










# https://stackoverflow.com/a/52714922
set(_refsizeinclude "${CMAKE_CURRENT_BINARY_DIR}/stage1.cmake")
set(_stage1generator "${CMAKE_SOURCE_DIR}/rosetta/bin/gen-stage1.py" )
# Python::Interpreter does not work?
#message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${_stage1generator}"
    "--builddir=${CMAKE_CURRENT_BINARY_DIR}"
    "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}"
    "--problemsizefile=${_problemsizefile}" 
    "--output=${_refsizeinclude}"
    "--configname=${ROSETTA_CONFIGNAME}'"
    ${ROSETTA_BENCH_FILTER}
  COMMAND_ERROR_IS_FATAL ANY
)

# Mark the input files as configure step dependencies so that the execute_process
# commands are retriggered on input file change.
#file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/generated/cmake/input_files _input_files)
# TODO: add_property
set_property(
  DIRECTORY APPEND PROPERTY 
    CMAKE_CONFIGURE_DEPENDS "${_problemsizefile};${_stage1generator}"
)

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
  get_property(_ppm TARGET "${_target}" PROPERTY ROSETTA_PPM)
  get_property(_basename TARGET "${_target}" PROPERTY ROSETTA_BASENAME)
  get_property(_benchpropfile TARGET "${_target}" PROPERTY ROSETTA_BENCHPROPFILE)
  get_property(_pbsize TARGET "${_target}" PROPERTY ROSETTA_PBSIZE)


  list(APPEND _benchmarks_run "register_benchmark(basename='${_basename}', ppm='${_ppm}', target='${_target}', exepath='$<TARGET_FILE:${_target}>', buildtype='$<CONFIG>', configname='${ROSETTA_CONFIGNAME}', benchpropfile='${_benchpropfile}', compiler='${CMAKE_CXX_COMPILER}', compilerflags='${CMAKE_CXX_FLAGS} ${_config_cxx_flags} $<TARGET_PROPERTY:${_target},COMPILE_OPTIONS> $<TARGET_PROPERTY:${_target},COMPILE_FLAGS>', pbsize=${_pbsize})")
endforeach ()
string(JOIN "\n" _benchmarks_run ${_benchmarks_run})


# TODO: Which configuration we want depends on the PPM. E.g for CUDA, $<COMPILE_LANGUAGE:CUDA> shouold be used for compiler flags
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchlist-$<CONFIG>$<IF:$<COMPILE_LANGUAGE:CXX>,,-$<COMPILE_LANGUAGE>>.py"
  CONTENT "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
  import sys
  sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')
from rosetta import runner_main,register_benchmark,rosetta_config
rosetta_config(resultsdir='${ROSETTA_RESULTS_DIR}')

${_benchmarks_run}

if __name__ == '__main__':
  runner_main()
"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

if (_default_build_type)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "benchlist-${_default_build_type}.py" "benchlist.py" 
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif ()



## Stage 2 #######################################################################################################

set(_benchpropfiles)
foreach (_benchmark IN LISTS _benchmarks)
  get_property(_benchpropfile TARGET "${_benchmark}" PROPERTY ROSETTA_BENCHPROPFILE)
  message("target ${_benchmark} penchprop ${_benchpropfile}")
  list(APPEND _benchpropfiles "${_benchpropfile}")
endforeach ( )
message("_benchpropfiles: ${_benchpropfiles}")

set(_stage2generator "${CMAKE_SOURCE_DIR}/rosetta/bin/gen-stage2.py" )
set(_stage2stampfile "stage2-stamp-$<CONFIG>")
add_custom_command(
    OUTPUT "${_stage2stampfile}" ${_benchpropfiles} 
    DEPENDS "${_stage2generator}"
    COMMAND Python::Interpreter "${_stage2generator}" 
      "--builddir=${CMAKE_CURRENT_BINARY_DIR}"
      "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}"
      "--resultsdir=${ROSETTA_RESULTS_DIR}"
      "--benchfile=${CMAKE_CURRENT_BINARY_DIR}/benchlist-$<CONFIG>.py"
      "--config=$<CONFIG>"
    COMMENT "Generate $<CONFIG> properties files ..."
  )

  add_custom_target(genfiles
    DEPENDS "${_stage2stampfile}" ${_benchpropfiles}
  )

  


##################################################################################################################



return()

get_property(_benchmarks_serial GLOBAL PROPERTY benchmarks_serial)
get_property(_references GLOBAL PROPERTY references)


# TODO: separate verifications lists should not be necessary
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


function (enable_ppm name shortname displayname)
  cmake_parse_arguments(_arg "" "" "BENCHMARKS"  ${ARGN})

  append_global_property(enabled_ppms "${name}")

  set(_benchmarks_run)
  foreach (_target IN LISTS _arg_BENCHMARKS)
    get_property(_reffile TARGET "${_target}" PROPERTY REFERENCE_FILE)
    get_property(_basename TARGET "${_target}" PROPERTY ROSETTA_BASENAME)
    list(APPEND _benchmarks_run "register_benchmark(basename='${_basename}', target='${_target}', exepath='$<TARGET_FILE:${_target}>', config='$<CONFIG>', ppm='${shortname}', configname='${ROSETTA_CONFIGNAME}')")
  endforeach ()
  list(JOIN _benchmarks_run "\n" _benchmarks_pylist)

  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/run-${name}-$<CONFIG>.py"
    CONTENT "#! /usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
  import sys
  sys.path.insert(0, '${CMAKE_SOURCE_DIR}/rosetta/lib')
from rosetta import runner_main,register_benchmark,rosetta_config
rosetta_config(resultsdir='${ROSETTA_RESULTS_DIR}')

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
      COMMAND Python::Interpreter "run-${name}.py" "--problemsizefile=${_problemsizefile}"
      COMMENT "Run ${displayname} benchmarks ..."
      USES_TERMINAL VERBATIM
    )
    add_custom_target("verify-${name}"
      COMMAND Python::Interpreter "run-${name}.py" "--problemsizefile=${_problemsizefile}" "--verify"
      COMMENT "Run ${displayname} correctness checks ..."
      USES_TERMINAL VERBATIM
    )
    add_dependencies("verify-${name}" "build-${name}")
  endif ()
endfunction()

if (ROSETTA_PPM_SERIAL)
  enable_ppm(serial serial Serial BENCHMARKS ${_benchmarks_serial})
endif ()

if (ROSETTA_PPM_CUDA)
  enable_ppm(cuda cuda CUDA BENCHMARKS ${_benchmarks_cuda})
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL)
  enable_ppm(openmp-parallel omp_parallel "OpenMP parallel" BENCHMARKS ${_benchmarks_openmp_parallel}) 
endif ()

if (ROSETTA_PPM_OPENMP_TASK)
  enable_ppm(openmp-task omp_task "OpenMP Tasks" BENCHMARKS ${_benchmarks_openmp_task})
endif ()

if (ROSETTA_PPM_OPENMP_TARGET)
  enable_ppm(openmp-target omp_target "OpenMP Target Offloading" BENCHMARKS ${_benchmarks_openmp_target})
endif ()



get_property(enabled_ppms GLOBAL PROPERTY enabled_ppms)

set(_ppms_pylist)
foreach (_name IN LISTS enabled_ppms)
  list(APPEND _ppms_pylist "load_register_file(thisscriptdir / 'run-${_name}-$<CONFIG>.py')")
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
import pathlib
from rosetta import runner_main, load_register_file
thisscriptdir = pathlib.Path(__file__).parent

${_ppms_pylist}

if __name__ == '__main__':
  runner_main()
"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )



foreach (_conftype IN LISTS _build_types)
  add_custom_target(run-${_conftype}
    DEPENDS ${_benchmarks}
    COMMAND  Python::Interpreter "run-${_conftype}.py" "--problemsizefile=${_problemsizefile}" "--bench"
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
    COMMAND Python::Interpreter "run.py" "--problemsizefile=${_problemsizefile}" "--bench"
    COMMENT "Run all benchmarks ..."
    USES_TERMINAL VERBATIM
  )
  add_dependencies(run build)

  add_custom_target(verify
    COMMAND Python::Interpreter "run.py" "--problemsizefile=${_problemsizefile}" "--verify"
    COMMENT "Verify all benchmarks ..."
    USES_TERMINAL VERBATIM
  )
  add_dependencies(verify build ${_references})
endif ()


