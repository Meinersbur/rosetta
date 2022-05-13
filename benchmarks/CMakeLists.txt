function (append_global_property propname)
  get_property(propval GLOBAL PROPERTY ${propname})
  list(APPEND propval ${ARGN})
  set_property(GLOBAL PROPERTY ${propname} "${propval}")
endfunction ()




function (add_benchmark_serial basename)
  if (NOT ROSETTA_PPM_SERIAL)
    return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.serial")
  set(_sources "${basename}.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries("${_target}" PRIVATE rosetta-serial)
  add_dependencies(build-serial "${_target}")

  append_global_property(benchmarks_serial "${_target}")
endfunction ()




function (add_benchmark_cuda basename)
  if (NOT ROSETTA_PPM_CUDA)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.cuda")
  set(_sources "${basename}.cu")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  target_link_libraries("${_target}" PRIVATE rosetta-cuda)
  add_dependencies(build-cuda "${_target}")

  append_global_property(benchmarks_cuda "${_target}")
endfunction ()





function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_parallel")
  set(_sources "${basename}.omp_parallel.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta-openmp OpenMP::OpenMP_CXX)
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  add_dependencies(build-openmp-parallel "${_target}")

  append_global_property(benchmarks_openmp_parallel "${_target}")
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_task")
  set(_sources "${basename}.omp_task.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta-openmp OpenMP::OpenMP_CXX)
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  add_dependencies(build-openmp-task "${_target}")

  append_global_property(benchmarks_openmp_task "${_target}")
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_N" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_target")
  set(_sources "${basename}.omp_target.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta-openmp OpenMP::OpenMP_CXX)
  if (_arg_DEFAULT_N)
    target_compile_definitions(${_target} PRIVATE "DEFAULT_N=${_arg_DEFAULT_N}")
  endif ()
  add_dependencies(build-openmp-target "${_target}")

  append_global_property(benchmarks_openmp_task "${_target}")
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
