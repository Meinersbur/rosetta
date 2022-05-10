

add_library(rosetta
  rosetta.cpp
  )
target_include_directories(rosetta PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(rosetta PUBLIC benchmark::benchmark)

if (ROSETTA_NO_EXCEPTIONS)
  # TODO: Introspection
  target_compile_options(rosetta PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
endif ()
