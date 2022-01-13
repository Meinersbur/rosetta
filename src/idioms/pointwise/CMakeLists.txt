add_executable(pointwise.seq.gbench pointwise.seq.cpp)
target_link_libraries(pointwise.seq.gbench PRIVATE benchmark::benchmark)
target_compile_options(pointwise.seq.gbench PUBLIC -fno-exceptions)


add_dependencies(gbench pointwise.seq.gbench )



