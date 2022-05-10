function (append_global_property propname)
  get_property(propval GLOBAL PROPERTY ${propname})
  list(APPEND propval ${ARGN})
  message("ARGN=${ARGN}")
  message("${propname}=${propval}")
  set_property(GLOBAL PROPERTY ${propname} "${propval}")
endfunction ()




function (add_benchmark_serial basename)
  message("add_benchmark_serial(${basename})")
  if (NOT ROSETTA_ENABLE_SERIAL)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "DEFAULT_SIZE" "SOURCES"  ${ARGN} )

  set(_target "${basename}.serial")
  set(_sources "${basename}.cpp")
  add_executable("${_target}" ${_sources} ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta)
  add_dependencies(gbench-serial "${_target}")

  append_global_property(benchmarks_serial "${_target}")
endfunction ()





function (add_benchmark_cuda basename)
  if (NOT ROSETTA_ENABLE_CUDA)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "" "SOURCES"  ${ARGN} )

  set(_target "${basename}.cuda")
  add_executable("${_target}" ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta)
  add_dependencies(gbench-cuda "${_target}")

  get_property(_tmp GLOBAL PROPERTY benchmarks_cuda)
  list(APPEND _tmp "${_target}")
  set_property(GLOBAL PROPERTY benchmarks_cuda "${_tmp}")
endfunction ()


function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_ENABLE_OPENMP_PARALLEL)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_parallel")
  add_executable("${_target}" ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta OpenMP::OpenMP_CXX)
  add_dependencies(gbench-openmp_parallel "${_target}")

  get_property(_tmp GLOBAL PROPERTY benchmarks_openmp_parallel)
  list(APPEND _tmp "${_target}")
  set_property(GLOBAL PROPERTY benchmarks_openmp_parallel "${_tmp}")
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_ENABLE_OPENMP_TASK)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_task")
  add_executable("${_target}" ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta OpenMP::OpenMP_CXX)
  add_dependencies(gbench-openmp_task "${_target}")

  get_property(_tmp GLOBAL PROPERTY benchmarks_openmp_task)
  list(APPEND _tmp "${_target}")
  set_property(GLOBAL PROPERTY benchmarks_openmp_task "${_tmp}")
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_ENABLE_OPENMP_TARGET)
      return ()
  endif ()
  cmake_parse_arguments(_arg "" "" "SOURCES"  ${ARGN} )

  set(_target "${basename}.openmp_target")
  add_executable("${_target}" ${_arg_SOURCES})
  target_link_libraries("${_target}" PRIVATE rosetta OpenMP::OpenMP_CXX)
  add_dependencies(gbench-openmp_target "${_target}")

  get_property(_tmp GLOBAL PROPERTY benchmarks_openmp_target)
  list(APPEND _tmp "${_target}")
  set_property(GLOBAL PROPERTY benchmarks_openmp_target "${_tmp}")
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
