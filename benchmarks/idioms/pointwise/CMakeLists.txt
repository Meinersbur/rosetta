add_benchmark_serial(pointwise DEFAULT_N "1024*1024")
add_benchmark_cuda(pointwise DEFAULT_N "1024*1024")

add_benchmark_openmp_parallel(pointwise)
add_benchmark_openmp_task(pointwise)
add_benchmark_openmp_target(pointwise)
