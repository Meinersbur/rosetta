
if (ROSETTA_ENABLE_SERIAL)
  add_executable(pointwise.seq.gbench pointwise.seq.cpp)
  target_link_libraries(pointwise.seq.gbench PRIVATE benchmark::benchmark)
  target_compile_options(pointwise.seq.gbench PUBLIC -fno-exceptions)
  add_dependencies(gbench-serial pointwise.seq.gbench)

  get_property(_tmp GLOBAL PROPERTY benchmarks_serial)
  list(APPEND _tmp "pointwise.seq.gbench")
  set_property(GLOBAL PROPERTY benchmarks_serial "${_tmp}")
endif()

if (ROSETTA_ENABLE_OPENMP_PARALLEL)
  add_executable(pointwise.omp_parallel.gbench pointwise.omp_parallel.cpp)
  target_link_libraries(pointwise.omp_parallel.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
  target_compile_options(pointwise.omp_parallel.gbench PUBLIC -fno-exceptions)
  add_dependencies(gbench-omp_parallel pointwise.omp_parallel.gbench)

  get_property(_tmp GLOBAL PROPERTY benchmarks_omp_parallel)
  list(APPEND _tmp "pointwise.omp_parallel.gbench")
  set_property(GLOBAL PROPERTY benchmarks_omp_parallel "${_tmp}")
endif ()

#add_executable(pointwise.omp_taskloop.gbench pointwise.omp_taskloop.cpp)
#target_link_libraries(pointwise.omp_taskloop.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
#target_compile_options(pointwise.omp_taskloop.gbench PUBLIC -fno-exceptions)
#add_dependencies(gbench pointwise.omp_taskloop.gbench)


#add_executable(pointwise.omp_target.gbench pointwise.omp_target.cpp)
#target_link_libraries(pointwise.omp_target.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
#target_compile_options(pointwise.omp_target.gbench PUBLIC -fno-exceptions)
#add_dependencies(gbench pointwise.omp_target.gbench)


if (ROSETTA_ENABLE_CUDA)
  add_executable(pointwise.cuda.gbench pointwise.cu)
  target_link_libraries(pointwise.cuda.gbench PRIVATE benchmark::benchmark)
  add_dependencies(gbench-cuda pointwise.cuda.gbench)

  get_property(_tmp GLOBAL PROPERTY benchmarks_cuda)
  list(APPEND _tmp "pointwise.cuda.gbench")
  set_property(GLOBAL PROPERTY benchmarks_cuda "${_tmp}")
endif ()
