


function (add_benchmark_serial basename)
  if (NOT ROSETTA_PPM_SERIAL)
    return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.serial.bench")
  set(_sources "${basename}.cpp")
  add_executable(${_target} ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries(${_target} PRIVATE rosetta-serial-bench)
  add_dependencies(build-serial "${_target}")

   # TODO: set_property(GLOBAL APPEND benchmarks_serial "${_target}")
  append_global_property(benchmarks_serial "${_target}")


  # TODO: unduplicate
  set(_verify_target "${basename}.serial.verify")
  add_executable("${_verify_target}" ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_verify_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries(${_verify_target} PRIVATE rosetta-serial-verify)
  add_dependencies(build-serial-verify "${_verify_target}")  # TODO: separate target for verifications

  append_global_property(verifications_serial "${_verify_target}")

  if (CMAKE_BUILD_TYPE)
    set(_reference_filename "${basename}.reference")
  else ()
    set(_reference_filename "${basename}.$<CONFIG>.reference")
  endif ()
  set(_reference "${CMAKE_CURRENT_BINARY_DIR}/${_reference_filename}")

  add_custom_command(
    OUTPUT "${_reference_filename}"
    DEPENDS "${_verify_target}"
    COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--gen-reference" "$<TARGET_FILE:${_verify_target}>" "${_reference}"
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

  set_property(TARGET ${_target}    
    PROPERTY VERIFY_TARGET "${_verify_target}"
  )
  set_property(TARGET ${_target}    
    PROPERTY VERIFY_FILE "$<TARGET_FILE:${_verify_target}>"
  )
  set_property(TARGET ${_target}    
    PROPERTY REFERENCE_FILE "${_reference}"
  )
endfunction ()








function (add_benchmark_cuda basename)
  if (NOT ROSETTA_PPM_CUDA)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )
  set(_sources "${basename}.cu")

  set(_target "${basename}.cuda.bench")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries("${_target}" PRIVATE rosetta-cuda)
  add_dependencies(build-cuda "${_target}")

  append_global_property(benchmarks_cuda "${_target}")


  set(_verify_target "${basename}.cuda.verify")
  add_executable("${_verify_target}" ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_verify_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries("${_verify_target}" PRIVATE rosetta-cuda-verify)
  add_dependencies(build-cuda-verify "${_verify_target}")
endfunction ()





function (add_benchmark_openmp_parallel_target basename _target _library)
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES" ${ARGN})

  set(_sources "${basename}.omp_parallel.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE ${_library})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()

  get_property(_reffile TARGET "${basename}.serial.verify" PROPERTY REFERENCE_FILE)
  set_property(TARGET ${_target}    
    PROPERTY REFERENCE_FILE "${_reffile}"
  )
endfunction ()

function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
      return ()
  endif ()

  set(_bench_target "${basename}.openmp_parallel.bench")
  set(_verify_target "${basename}.openmp_parallel.verify")
  add_benchmark_openmp_parallel_target(${basename} ${_bench_target} rosetta-openmp-bench ${ARGN})
  add_dependencies(build-openmp-parallel "${_bench_target}")
  append_global_property(benchmarks_openmp_parallel "${_bench_target}")
  set_property(TARGET ${_bench_target}    
    PROPERTY VERIFY_TARGET "${_verify_target}"
  )

  add_benchmark_openmp_parallel_target(${basename} ${_verify_target} rosetta-openmp-verify ${ARGN})
  add_dependencies(build-openmp-parallel-verify "${_verify_target}")
  append_global_property(verifications_openmp_parallel "${_verify_target}")
endfunction ()






function (add_benchmark_openmp_task_target basename _target _library)
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES" ${ARGN})

  set(_sources "${basename}.omp_task.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE ${_library})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()

  get_property(_reffile TARGET "${basename}.serial.verify" PROPERTY REFERENCE_FILE)
  set_property(TARGET ${_target}    
    PROPERTY REFERENCE_FILE "${_reffile}"
  )
endfunction ()

function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  
  set(_bench_target "${basename}.openmp_task.bench")
  set(_verify_target "${basename}.openmp_task.verify")
  add_benchmark_openmp_task_target(${basename} ${_bench_target} rosetta-openmp-bench ${ARGN})
  add_dependencies(build-openmp-task "${_bench_target}")
  append_global_property(benchmarks_openmp_task "${_bench_target}")
  set_property(TARGET ${_bench_target}    
    PROPERTY VERIFY_TARGET "${_verify_target}"
  )

  add_benchmark_openmp_task_target(${basename} ${_verify_target} rosetta-openmp-verify ${ARGN})
  add_dependencies(build-openmp-task-verify "${_verify_target}")
  append_global_property(verifications_openmp_task "${_verify_target}")
endfunction ()



function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_target")
  set(_sources "${basename}.omp_target.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta-openmp-target)
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  add_dependencies(build-openmp-target "${_target}")

  append_global_property(benchmarks_openmp_target "${_target}")
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
