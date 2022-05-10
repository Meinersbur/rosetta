add_test_serial(pointwise)



add_executable(pointwise.omp_parallel.gbench pointwise.omp_parallel.cpp)
target_link_libraries(pointwise.omp_parallel.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(pointwise.omp_parallel.gbench PUBLIC -fno-exceptions)
add_dependencies(gbench pointwise.omp_parallel.gbench)


add_executable(pointwise.omp_taskloop.gbench pointwise.omp_taskloop.cpp)
target_link_libraries(pointwise.omp_taskloop.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(pointwise.omp_taskloop.gbench PUBLIC -fno-exceptions)
add_dependencies(gbench pointwise.omp_taskloop.gbench)


add_executable(pointwise.omp_target.gbench pointwise.omp_target.cpp)
target_link_libraries(pointwise.omp_target.gbench PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(pointwise.omp_target.gbench PUBLIC -fno-exceptions)
add_dependencies(gbench pointwise.omp_target.gbench)


add_test_cuda(pointwise)

