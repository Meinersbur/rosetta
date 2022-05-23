
if (ROSETTA_PPM_SERIAL)
  add_library(rosetta-serial
    rosetta-serial.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-serial PUBLIC ROSETTA_SERIAL=1)
  target_include_directories(rosetta-serial PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #target_link_libraries(rosetta-serial PUBLIC benchmark::benchmark)


  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-serial PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()
endif ()


if (ROSETTA_PPM_CUDA)
  add_library(rosetta-cuda
    rosetta-cuda.cu
    rosetta-common.cpp
    rosetta-nvidia.cpp
    )
    target_compile_definitions(rosetta-cuda PUBLIC ROSETTA_PPM_CUDA=1)
  target_include_directories(rosetta-cuda PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #target_link_libraries(rosetta-cuda PUBLIC benchmark::benchmark)
  target_link_libraries(rosetta-cuda PUBLIC CUDA::cudart)
  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-cuda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()
endif ()


if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK OR ROSETTA_PPM_OPENMP_TARGET)
  add_library(rosetta-openmp
    rosetta-openmp.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-openmp PUBLIC ROSETTA_SERIAL=1)
  target_include_directories(rosetta-openmp PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #target_link_libraries(rosetta-openmp PUBLIC benchmark::benchmark)
  if (ROSETTA_PLATFORM_NVIDIA)
  target_sources(rosetta-openmp PRIVATE rosetta-nvidia.cpp)
  endif ()
endif ()
