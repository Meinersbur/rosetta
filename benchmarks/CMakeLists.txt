




## Stage 1 #######################################################################################################

# internal
# TODO: basename is not required
function (add_benchmark_ppm basename ppm ppm_shortname _library)
  cmake_parse_arguments(_arg "" "" "SOURCES" ${ARGN})

  file(RELATIVE_PATH _relpath  "${CMAKE_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}")
  string(REPLACE "/" "." _relpath_dots "${_relpath}")
  #set(basename "${_relpath_dots}")
  set(benchname "${basename}")

  set(_benchprop_filename "${benchname}.$<CONFIG>.benchprop.cxx")
  #configure_file("config.h.in" "${_benchprop_filename}")

  set(_genconfig_target "genconfig-${benchname}")
  if (NOT TARGET "${_genconfig_target}")
    add_custom_command(
      OUTPUT "${_benchprop_filename}"
      DEPENDS "${_problemsizefile}" "${GENCONFIG_EXECUTABLE}"
      COMMAND Python::Interpreter "${GENCONFIG_EXECUTABLE}" 
        "--benchname=${benchname}"
        "--problemsizefile=${_problemsizefile}"
        "--output=${_benchprop_filename}"
        "--resultsdir=${ROSETTA_RESULTS_DIR}"
        "--compiler=${CMAKE_CXX_COMPILER}" # FIXME: Depends on PPM
        "--compiler-flags=${CMAKE_CXX_FLAGS}" # FIXME: Not complete and depends on configuration and PPM
        "--configname=${ROSETTA_CONFIGNAME}"
        "--config=$<CONFIG>"
      COMMENT "Generate ${benchname} $<CONFIG> properties file ..."
    )
    add_custom_target("${_genconfig_target}"
      DEPENDS "${_benchprop_filename}"
    )
    set_target_properties("${_genconfig_target}" PROPERTIES  FOLDER "Benchmarks/Genconfig")
  endif ()


   


  set(_bench_target "${basename}.${ppm_shortname}")
  set(_sources ${_arg_SOURCES} "${_benchprop_filename}")

  string(REPLACE "-" "_" ppm_mixed "${ppm}")

  add_executable("${_bench_target}" ${_sources})
  message("added target: ${_bench_target}")
  set_target_properties("${_bench_target}" 
    PROPERTIES 
      FOLDER "Benchmarks/${ppm}"
      ROSETTA_BASENAME "${basename}"
    )
  target_link_libraries("${_bench_target}" PRIVATE "${_library}")
  if (ROSETTA_REALTYPE)
    # TODO: Support compiling mutiple real types at the same time?
    target_compile_definitions(${_bench_target} PRIVATE "ROSETTA_REALTYPE=${ROSETTA_REALTYPE}")
  endif ()


  get_property(_reffile TARGET "${basename}.serial" PROPERTY REFERENCE_FILE)


  set_property(TARGET ${_bench_target} PROPERTY REFERENCE_FILE "${_reffile}")
  #set_property(TARGET ${_verify_target} PROPERTY REFERENCE_FILE "${_reffile}")
  set_property(TARGET ${_bench_target} PROPERTY VERIFY_TARGET "${_verify_target}")

  append_global_property("benchmarks_${ppm_mixed}" "${_bench_target}")
  #append_global_property("verifications_${ppm_mixed}" "${_verify_target}")


  #add_custom_target("${basename}.${ppm}")
  #add_dependencies("${basename}.${ppm}" "${_bench_target}" "${_verify_target}")

  add_dependencies("build-${ppm}" "${_bench_target}")
  #add_dependencies("build-${ppm}-verify" "${_verify_target}")
endfunction ()




function (make_reference_output basename ppm ppm_shortname _library)
  set(_reference_dir "${PROJECT_BINARY_DIR}/references")
  #if (CMAKE_BUILD_TYPE)
    set(_reference_filename "${basename}.${ROSETTA_PROBLEMSIZE}.reference_output")
  #else ()
  #  set(_reference_filename "${basename}.$<CONFIG>.reference")
  #endif ()
  set(_reffile "${_reference_dir}/${_reference_filename}")
  set(_bench_target "${basename}.${ppm_shortname}")

  # TODO: different reference file for each problemsize? Hardcode problemsize for verification?
  #add_custom_command(
  #  OUTPUT "${_reference_filename}"
  #  DEPENDS "${_bench_target}" "${RUNNER_EXECUTABLE}"
  #  COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--problemsizefile=${_problemsizefile}" "--gen-reference" "$<TARGET_FILE:${_bench_target}>" "${_reffile}"
  #  COMMENT "Generate ${basename} reference output (using serial output) ..."
  #)
  add_custom_command(
    OUTPUT "${_reffile}"
    DEPENDS "${_bench_target}" 
    COMMAND "${_bench_target}" --verify --verifyfile "${_reffile}"
    COMMENT "Generate ${basename} reference output (using serial output)..."
  )

  add_custom_target("reference-${basename}"
    DEPENDS "${_reffile}"
  )
  append_global_property(references "reference-${basename}")
  set_property(TARGET ${_verify_target}    
    PROPERTY REFERENCE_FILE "${_reffile}"
  )
  append_global_property(references_serial "${_reference}")
endfunction()



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
  add_benchmark_ppm("${basename}" "cuda" "cuda" "rosetta-cuda" ${ARGN} SOURCES "${basename}.cu")
endfunction ()


function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-parallel" "omp_parallel" "rosetta-openmp-parallel" ${ARGN} SOURCES "${basename}.omp_parallel.cxx")
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-task" "omp_task" "rosetta-openmp-task" ${ARGN} SOURCES "${basename}.omp_task.cxx")
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-target" "omp_target" "rosetta-openmp-target" ${ARGN} SOURCES "${basename}.omp_target.cxx")
endfunction ()







function (add_benchmark basename)
  #cmake_parse_arguments(ARG "" "NAME" "SERIAL;CUDA;OMP_PARALLEL;OMP_TASK;OMP_TARGET" ${ARGN})
  message("add_benchmark(${basename})")

  set(_source_serial "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cpp")
  if (EXISTS "${_source_serial}")
    add_benchmark_serial("${basename}" SOURCES ${_source_serial})
  endif ()

  set(_source_cuda "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cu")
  if (EXISTS "${_source_cuda}")
    add_benchmark_cuda("${basename}" SOURCES ${_source_cuda})
  endif ()

  set(_source_openmp_parallel "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_parallel.cpp")
  if (EXISTS "${_source_cuda}")
    add_benchmark_openmp_parallel("${basename}" SOURCES ${_source_openmp_parallel})
  endif ()

  set(_source_openmp_task "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_task.cpp")
  if (EXISTS "${_source_openmp_task}")
    add_benchmark_openmp_task("${basename}" SOURCES ${_source_openmp_task})
  endif ()

  set(_source_openmp_target "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_target.cpp")
  if (EXISTS "${_source_openmp_target}")
    add_benchmark_openmp_target("${basename}" SOURCES ${_source_openmp_target})
  endif ()
endfunction ()





#add_subdirectory(idioms)
#add_subdirectory(features)
#add_subdirectory(synth)
#add_subdirectory(suites)
#add_subdirectory(apps)






# https://stackoverflow.com/a/52714922
set(_refsizeinclude "${CMAKE_CURRENT_BINARY_DIR}/refsizes.cmake")
# Python::Interpreter does not work?
#message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/rosetta/bin/gen-stage1.py" 
    "--problemsizefile=${_problemsizefile}" 
    "--output=${_refsizeinclude}"
    "--benchdir=${CMAKE_CURRENT_SOURCE_DIR}"
    "--builddir=${CMAKE_CURRENT_BINARY_DIR}"
    "--configname=${ROSETTA_CONFIGNAME}'"
  COMMAND_ERROR_IS_FATAL ANY
)

# Mark the input files as configure step dependencies so that the execute_process
# commands are retriggered on input file change.
#file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/generated/cmake/input_files _input_files)
# TODO: add_property
set_property(
  DIRECTORY APPEND PROPERTY 
    CMAKE_CONFIGURE_DEPENDS "${_problemsizefile};${CMAKE_SOURCE_DIR}/rosetta/bin/genrefsizes.py"
)

# Add the generated CMake targets.
include("${_refsizeinclude}")




## Stage 2 #######################################################################################################


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


