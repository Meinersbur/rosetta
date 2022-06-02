
function(enable_platform_nvidia target)
  if (NOT ROSETTA_PLATFORM_NVIDIA)
    return ()
  endif ()

  target_sources("${target}" PRIVATE rosetta-nvidia.cpp)
  target_compile_definitions("${target}" PUBLIC ROSETTA_PLATFORM_NVIDIA=1)
  target_link_libraries("${target}" PUBLIC CUDA::cudart CUDA::cupti CUDA::cuda_driver)
  message("CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories("${target}" PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories("${target}" PUBLIC "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include") # For unknown reasons on JLSE
endfunction()



if (ROSETTA_PPM_SERIAL)
  add_library(rosetta-serial-bench
    rosetta-serial.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-serial-bench PUBLIC ROSETTA_SERIAL=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-serial-bench PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")

  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-serial-bench PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()

  # TODO: un-duplicate
  add_library(rosetta-serial-verify
    rosetta-serial.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-serial-verify PUBLIC ROSETTA_SERIAL=1 ROSETTA_VERIFY=1)
  target_include_directories(rosetta-serial-verify PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")

  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-serial-verify PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()
endif ()


if (ROSETTA_PPM_CUDA)
  add_library(rosetta-cuda
    rosetta-cuda.cu
    rosetta-common.cpp
    rosetta-nvidia.cpp
    )
    target_compile_definitions(rosetta-cuda PUBLIC ROSETTA_PPM_CUDA=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-cuda PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #target_link_libraries(rosetta-cuda PUBLIC benchmark::benchmark)
  target_link_libraries(rosetta-cuda PUBLIC CUDA::cudart)
  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-cuda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()

  enable_platform_nvidia(rosetta-cuda)
endif ()



if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK)
  add_library(rosetta-openmp
    rosetta-openmp.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-openmp PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-openmp PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries( rosetta-openmp PUBLIC OpenMP::OpenMP_CXX)
endif ()


if (ROSETTA_PPM_OPENMP_TARGET)
  add_library(rosetta-openmp-target
    rosetta-openmp.cpp
    rosetta-common.cpp
  )
  target_compile_definitions(rosetta-openmp-target PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_TARGET=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-openmp-target PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-target PUBLIC OpenMP::OpenMP_Offload_CXX)
  enable_platform_nvidia(rosetta-openmp-target)
endif ()
