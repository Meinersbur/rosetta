# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang") #don't need to check
# message(STATUS "Clang Compiler found: ${CMAKE_CXX_COMPILER}")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
if(COMPILER_SUPPORTS_FSYCL)
  message(STATUS "Compiler supports -fsycl flag")
  find_package(LLVM CONFIG QUIET)
  if(LLVM_FOUND)
    set(LLVMSYCL_FOUND TRUE)
    set(SYCL_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}/sycl")
    #add_compile_options("-fsycl")
    if(ROSETTA_PLATFORM_NVIDIA)
      set(SYCL_CXX_FLAGS
      "-fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
      CACHE STRING "Compiler arguments for SYCL on Nvidia")
      #add_compile_options("-fsycl-targets=nvptx64-nvidia-cuda-sycldevice")
    endif()
    else()
      set(SYCL_CXX_FLAGS "-fsycl"
      CACHE STRING "Compiler arguments for SYCL")
    endif()
    message(
      STATUS
        "LLVM Compiler found: include dir: ${LLVM_INCLUDE_DIRS}. SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}"
    )
  else()
    message(STATUS "LLVM Compiler NOT found")
  endif()
else()
  message(STATUS "Compiler does not support -fsycl flag")
endif()
# endif()
