set(_common_sources
    "rosetta-common.cpp"
    "rosetta-stat.cpp"
    "cdflib.cpp"
    "stdtr.c"
    "ndtr.c"
    "ndtri.c"
    "incbet.c"
    "incbi.c"
    "mtherr.c"
    "const.c"
    "gamma.c"
    "polevl.c"
    "igam.c")

function (enable_platform_nvidia target)
  if (NOT ROSETTA_PLATFORM_NVIDIA)
    return()
  endif ()

  target_sources("${target}" PRIVATE rosetta-nvidia.cpp)
  target_compile_definitions("${target}" PUBLIC ROSETTA_PLATFORM_NVIDIA=1)
  target_link_libraries("${target}" PUBLIC CUDA::cudart CUDA::cupti CUDA::cuda_driver)
  # message("CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories("${target}" PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories("${target}" PUBLIC "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include"
  )# For unknown reasons on JLSE

  # set_target_properties("${target}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction ()

if (ROSETTA_PPM_SERIAL)
  add_library(rosetta-serial rosetta-serial.cpp ${_common_sources})
  set_target_properties(rosetta-serial PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-serial PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-serial PUBLIC ROSETTA_PPM_SERIAL=1)
  target_include_directories(rosetta-serial PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  set_property(TARGET rosetta-serial PROPERTY POSITION_INDEPENDENT_CODE ON) # Because of -fPIE
  if (HAS_INCLUDE_CHARCONV_FLOAT)
    target_compile_definitions(rosetta-serial PUBLIC HAS_INCLUDE_CHARCONV_FLOAT=1)
  endif ()

  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-serial PUBLIC $<$<COMPILE_LANGUAGE:CXX,CUDA>:-fno-exceptions>)
  endif ()
endif ()

if (ROSETTA_PPM_CUDA)
  add_library(rosetta-cuda rosetta-cuda.cu rosetta-nvidia.cpp ${_common_sources})
  set_target_properties(rosetta-cuda PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-cuda PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-cuda PUBLIC ROSETTA_PPM_CUDA=1)
  target_include_directories(rosetta-cuda PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #target_link_libraries(rosetta-cuda PUBLIC CUDA::cudart) # Requires find_package(CUDAToolkit)
  if (ROSETTA_NO_EXCEPTIONS)
    # TODO: Introspection
    target_compile_options(rosetta-cuda PUBLIC $<$<COMPILE_LANGUAGE:CXX,CUDA>:-fno-exceptions>)
  endif ()
  target_compile_options(
    rosetta-cuda INTERFACE "-fPIE")

  # WORKAROUND:  relocation R_X86_64_32 against `.rodata.str1.1' can not be used when making a PIE object; recompile with -fPIE (clang enforces -pie by default)
  set_property(TARGET rosetta-cuda PROPERTY POSITION_INDEPENDENT_CODE ON
  )# Should be CMake's way to avoid above error, but doesn't work
  target_link_libraries(rosetta-cuda PUBLIC CUDA::cudart)

  enable_platform_nvidia(rosetta-cuda)
endif ()

if (ROSETTA_PPM_OPENMP_PARALLEL)
  add_library(rosetta-openmp-parallel rosetta-openmp.cpp ${_common_sources})
  set_target_properties(rosetta-openmp-parallel PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-openmp-parallel PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-openmp-parallel PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_PARALLEL=1)
  target_include_directories(rosetta-openmp-parallel PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-parallel PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  #set_property(TARGET rosetta-openmp-parallel PROPERTY POSITION_INDEPENDENT_CODE ON) # Because of -fPIE
endif ()

if (ROSETTA_PPM_OPENMP_TASK)
  add_library(rosetta-openmp-task rosetta-openmp.cpp ${_common_sources})
  set_target_properties(rosetta-openmp-task PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-openmp-task PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-openmp-task PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_TASK=1)
  target_include_directories(rosetta-openmp-task PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-task PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  #set_property(TARGET rosetta-openmp-task PROPERTY POSITION_INDEPENDENT_CODE ON) # Because of -fPIE
endif ()

if (ROSETTA_PPM_OPENMP_TARGET)
  add_library(rosetta-openmp-target rosetta-openmp.cpp ${_common_sources})
  set_target_properties(rosetta-openmp-target PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-openmp-target PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-openmp-target PUBLIC ROSETTA_PPM_OPENMP=1 ROSETTA_PPM_OPENMP_TARGET=1)
  target_include_directories(rosetta-openmp-target PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  target_link_libraries(rosetta-openmp-target PUBLIC OpenMP::OpenMP_Offload_C OpenMP::OpenMP_Offload_CXX)
  #set_property(TARGET rosetta-openmp-target PROPERTY POSITION_INDEPENDENT_CODE ON) # Because of -fPIE
  enable_platform_nvidia(rosetta-openmp-target)
endif ()

if (ROSETTA_PPM_SYCL)
  add_library(rosetta-sycl rosetta-sycl.cpp ${_common_sources})
  set_target_properties(rosetta-sycl PROPERTIES FOLDER "Runtime")
  target_compile_features(rosetta-sycl PUBLIC cxx_std_17)
  target_compile_definitions(rosetta-sycl PUBLIC ROSETTA_PPM_SYCL=1)
  target_include_directories(rosetta-sycl PUBLIC "${CMAKE_SOURCE_DIR}/rosetta/include")
  #https://intel.github.io/llvm-docs/GetStartedGuide.html
  target_link_options(rosetta-sycl PUBLIC -fsycl)
  #target_compile_options(rosetta-sycl PUBLIC -fsycl)
  target_link_libraries(rosetta-sycl PUBLIC IntelSYCL::SYCL_CXX)
endif ()
