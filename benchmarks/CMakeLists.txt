

# internal
function (add_benchmark_ppm basename ppm ppm_shortname _library)
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES" ${ARGN})

  set(_bench_target "${basename}.${ppm_shortname}.bench")
  set(_verify_target "${basename}.${ppm_shortname}.verify")
  set(_sources ${_arg_SOURCES})


  string(REPLACE "-" "_" ppm_mixed "${ppm}")

  add_executable("${_bench_target}" ${_sources})
  add_executable("${_verify_target}" ${_sources})
  target_link_libraries("${_bench_target}" PRIVATE "${_library}-bench")
  target_link_libraries("${_verify_target}" PRIVATE "${_library}-verify")
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_bench_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
    target_compile_definitions(${_verify_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()

  if (ppm STREQUAL "serial")
    if (CMAKE_BUILD_TYPE)
      set(_reference_filename "${basename}.reference")
    else ()
      set(_reference_filename "${basename}.$<CONFIG>.reference")
    endif ()
    set(_reffile "${CMAKE_CURRENT_BINARY_DIR}/${_reference_filename}")

    # TODO: different reference file for reach problemsize? Hardcode problemsize for verification?
    add_custom_command(
      OUTPUT "${_reference_filename}"
      DEPENDS "${_verify_target}"
      COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--problemsizefile=${_problemsizefile}" "--gen-reference" "$<TARGET_FILE:${_verify_target}>" "${_reffile}"
      COMMENT "Generate ${basename} reference output (using serial output) ..."
    )
    add_custom_target("reference-${basename}"
      DEPENDS "${_reference_filename}"
    )
    append_global_property(references "reference-${basename}")
    set_property(TARGET ${_verify_target}    
      PROPERTY REFERENCE_FILE "${_reference}"
    )
    append_global_property(references_serial "${_reference}")
  else ()
    get_property(_reffile TARGET "${basename}.serial.verify" PROPERTY REFERENCE_FILE)
  endif ()
  set_property(TARGET ${_bench_target} PROPERTY REFERENCE_FILE "${_reffile}")
  set_property(TARGET ${_verify_target} PROPERTY REFERENCE_FILE "${_reffile}")
  set_property(TARGET ${_bench_target} PROPERTY VERIFY_TARGET "${_verify_target}")

  append_global_property("benchmarks_${ppm_mixed}" "${_bench_target}")
  append_global_property("verifications_${ppm_mixed}" "${_verify_target}")


  add_custom_target("${basename}.${ppm}")
  add_dependencies("${basename}.${ppm}" "${_bench_target}" "${_verify_target}")

  add_dependencies("build-${ppm}" "${_bench_target}")
  add_dependencies("build-${ppm}-verify" "${_verify_target}")
endfunction ()





function (add_benchmark_serial basename)
  if (NOT ROSETTA_PPM_SERIAL)
    return ()
  endif ()
  #cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES" ${ARGN})

  add_benchmark_ppm("${basename}" "serial" "serial" "rosetta-serial" ${ARGN} SOURCES "${basename}.cxx")

  #set(_target "${basename}.serial.bench")
  #set(_sources "${basename}.cxx")
  #add_executable(${_target} ${_sources} ${_arg_SOURCES})
  #if (_arg_DEFAULT_N)
  #  target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  #endif ()
  #target_link_libraries(${_target} PRIVATE rosetta-serial-bench)
  #add_dependencies(build-serial "${_target}")

   # TODO: set_property(GLOBAL APPEND benchmarks_serial "${_target}")
  #append_global_property(benchmarks_serial "${_target}")


  # TODO: unduplicate
  #set(_verify_target "${basename}.serial.verify")
  #add_executable("${_verify_target}" ${_sources} ${_arg_SOURCES})
  #if (_arg_DEFAULT_N)
  #  target_compile_definitions(${_verify_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  #endif ()
  #target_link_libraries(${_verify_target} PRIVATE rosetta-serial-verify)
  #add_dependencies(build-serial-verify "${_verify_target}")  # TODO: separate target for verifications

  #append_global_property(verifications_serial "${_verify_target}")


 
 

  #set_property(TARGET ${_target}    
  #  PROPERTY VERIFY_TARGET "${_verify_target}"
  #)
  #set_property(TARGET ${_target}    
  #  PROPERTY VERIFY_FILE "$<TARGET_FILE:${_verify_target}>"
  #)
  #set_property(TARGET ${_target}    
  #  PROPERTY REFERENCE_FILE "${_reference}"
  #)

  #add_custom_target("${basename}.serial")
  #add_dependencies("${basename}.serial" "${_target}" "${_verify_target}")
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
  add_benchmark_ppm("${basename}" "openmp-parallel" "omp_parallel" "rosetta-openmp" ${ARGN} SOURCES "${basename}.omp_parallel.cxx")
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-task" "omp_task" "rosetta-openmp" ${ARGN} SOURCES "${basename}.omp_task.cxx")
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-target" "omp_target" "rosetta-openmp-target" ${ARGN} SOURCES "${basename}.omp_target.cxx")
endfunction ()







function (add_benchmark basename)
  #cmake_parse_arguments(ARG "" "NAME" "SERIAL;CUDA;OMP_PARALLEL;OMP_TASK;OMP_TARGET" ${ARGN})

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





add_subdirectory(idioms)
#add_subdirectory(synth)
#add_subdirectory(suites)
#add_subdirectory(apps)
