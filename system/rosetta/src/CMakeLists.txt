add_library(rosetta-serial
  rosetta-serial.cpp
  rosetta-common.cpp
  )
target_compile_definitions(rosetta-serial PUBLIC ROSETTA_SERIAL=1)
target_include_directories(rosetta-serial PUBLIC "${CMAKE_SOURCE_DIR}/system/rosetta/include")
target_link_libraries(rosetta-serial PUBLIC benchmark::benchmark)


if (ROSETTA_NO_EXCEPTIONS)
  # TODO: Introspection
  target_compile_options(rosetta-serial PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
endif ()


add_library(rosetta-cuda
  rosetta-cuda.cu
  rosetta-common.cpp
  )
  target_compile_definitions(rosetta-cuda PUBLIC ROSETTA_CUDA=1)
target_include_directories(rosetta-cuda PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(rosetta-cuda PUBLIC benchmark::benchmark)
if (ROSETTA_NO_EXCEPTIONS)
  # TODO: Introspection
  target_compile_options(rosetta-cuda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
endif ()


add_library(rosetta-openmp
  rosetta-openmp.cpp
  rosetta-common.cpp
  )
target_compile_definitions(rosetta-openmp PUBLIC ROSETTA_SERIAL=1)
target_include_directories(rosetta-openmp PUBLIC "${CMAKE_SOURCE_DIR}/system/rosetta/include")
target_link_libraries(rosetta-openmp PUBLIC benchmark::benchmark)
