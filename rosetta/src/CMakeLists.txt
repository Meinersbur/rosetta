
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
  add_library(rosetta-cuda-bench
    rosetta-cuda.cu
    rosetta-common.cpp
    rosetta-nvidia.cpp
    )
  target_compile_definitions(rosetta-cuda-bench PUBLIC ROSETTA_PPM_CUDA=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-cuda-bench PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-cuda-bench PUBLIC CUDA::cudart)
  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-cuda-bench PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()
  enable_platform_nvidia(rosetta-cuda-bench)


  # TODO: un-duplicate
  add_library(rosetta-cuda-verify
    rosetta-cuda.cu
    rosetta-common.cpp
    rosetta-nvidia.cpp
    )
  target_compile_definitions(rosetta-cuda-verify PUBLIC ROSETTA_PPM_CUDA=1 ROSETTA_VERIFY=1)
  target_include_directories(rosetta-cuda-verify PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-cuda-verify PUBLIC CUDA::cudart)
  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-cuda-verify PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>)
  endif ()
  enable_platform_nvidia(rosetta-cuda-verify)
endif ()



if (ROSETTA_PPM_OPENMP_PARALLEL OR ROSETTA_PPM_OPENMP_TASK)
  add_library(rosetta-openmp-bench
    rosetta-openmp.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-openmp-bench PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-openmp-bench PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-bench PUBLIC OpenMP::OpenMP_CXX)


  # TODO: un-duplicate
  add_library(rosetta-openmp-verify
    rosetta-openmp.cpp
    rosetta-common.cpp
    )
  target_compile_definitions(rosetta-openmp-verify PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_VERIFY=1)
  target_include_directories(rosetta-openmp-verify PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-verify PUBLIC OpenMP::OpenMP_CXX)
endif ()


if (ROSETTA_PPM_OPENMP_TARGET)
  add_library(rosetta-openmp-target-bench
    rosetta-openmp.cpp
    rosetta-common.cpp
  )
  target_compile_definitions(rosetta-openmp-target-bench PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_TARGET=1 ROSETTA_BENCH=1)
  target_include_directories(rosetta-openmp-target-bench PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-target-bench PUBLIC OpenMP::OpenMP_Offload_CXX)
  enable_platform_nvidia(rosetta-openmp-target-bench)


  # TODO: un-duplicate
  add_library(rosetta-openmp-target-verify
    rosetta-openmp.cpp
    rosetta-common.cpp
  )
  target_compile_definitions(rosetta-openmp-target-verify PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_TARGET=1 ROSETTA_VERIFY=1)
  target_include_directories(rosetta-openmp-target-verify PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-target-verify PUBLIC OpenMP::OpenMP_Offload_CXX)
  enable_platform_nvidia(rosetta-openmp-target-verify)
endif ()
