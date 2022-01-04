add_executable(pointwise.seq pointwise.seq.cpp)
target_link_libraries(pointwise.seq PRIVATE benchmark::benchmark)
target_compile_options(pointwise.seq PUBLIC -fno-exceptions)
