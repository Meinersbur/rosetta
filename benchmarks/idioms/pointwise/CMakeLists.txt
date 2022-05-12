add_benchmark_serial(pointwise DEFAULT_N "1024*1024")
add_benchmark_cuda(pointwise DEFAULT_N "1024*1024")

add_benchmark_openmp_parallel(pointwise DEFAULT_N "1024*1024")
add_benchmark_openmp_task(pointwise DEFAULT_N "1024*1024")
add_benchmark_openmp_target(pointwise DEFAULT_N "1024*1024")
