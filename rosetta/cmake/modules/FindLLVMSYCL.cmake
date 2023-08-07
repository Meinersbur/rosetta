if(CMAKE_CXX_COMPILER_ID MATCHES "Clang") # don't need to check
  message(STATUS "Clang Compiler found: ${CMAKE_CXX_COMPILER}")
  include(CheckCXXCompilerFlag)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_LINK_OPTIONS)
  set(CMAKE_REQUIRED_LIBRARIES)
  check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
  if(COMPILER_SUPPORTS_FSYCL)
    message(STATUS "Compiler supports -fsycl flag")
    find_package(LLVM CONFIG QUIET)
    if(LLVM_FOUND)
      set(LLVMSYCL_FOUND TRUE)
      set(SYCL_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}/sycl")
      if(CMAKE_CUDA_COMPILER)
        set(SYCL_FLAGS
            "-fsycl;-fsycl-targets=nvptx64-nvidia-cuda"
            CACHE STRING "Compiler arguments for SYCL on Nvidia")
        separate_arguments(_SYCL_CXX_FLAGS "${SYCL_FLAGS}" NATIVE_COMMAND)
        set(SYCL_CXX_FLAGS ${_SYCL_CXX_FLAGS})

        message(
          "Set SYCL flag: -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice")
      else()
        set(SYCL_FLAGS
            "-fsycl"
            CACHE STRING "Compiler arguments for SYCL on Nvidia")
        separate_arguments(_SYCL_CXX_FLAGS "${SYCL_FLAGS}" NATIVE_COMMAND)
        set(SYCL_CXX_FLAGS ${_SYCL_CXX_FLAGS})
        message("Set SYCL flag: ${SYCL_CXX_FLAGS}")
      endif()
      message(
        STATUS
          "LLVM Compiler found: include dir: ${LLVM_INCLUDE_DIRS}. SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}"
      )
    else()
      message(STATUS "LLVM Compiler NOT found")
    endif()

    add_library(SYCL::SYCL_CXX INTERFACE IMPORTED)
    set_property(TARGET SYCL::SYCL_CXX PROPERTY INTERFACE_COMPILE_OPTIONS
                                                ${SYCL_CXX_FLAGS})
    set_property(TARGET SYCL::SYCL_CXX PROPERTY INTERFACE_LINK_OPTIONS
                                                ${SYCL_CXX_FLAGS})

  else()
    message(STATUS "Compiler does not support -fsycl flag")
  endif()
endif()
