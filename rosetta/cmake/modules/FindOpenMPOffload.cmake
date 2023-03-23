# find module https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules

# Inputs: OpenMPOffload_FIND_VERSION OpenMPOffload_FIND_QUIETLY OpenMPOffload_FIND_REQUIRED

# Outputs: OpenMPOffload_FOUND OpenMPOffload_DEFINITIONS OpenMPOffload import target

if (NOT DEFINED OpenMP_FOUND)
  if (OpenMPOffload_FIND_REQUIRED)
    find_package(OpenMP REQUIRED)
  else ()
    find_package(OpenMP OPTIONAL)
  endif ()
endif ()

include(CheckCXXSourceCompiles)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Ubuntu: "GCC is not configured to support default as offload target"
  #set(OPENMP_OFFLOADING_CFLAGS  "-foffload=default" CACHE STRING "Compiler arguments for OpenMP offloading")
  #set(OPENMP_OFFLOADING_LDFLAGS "-foffload=default" CACHE STRING "Linker arguments for OpenMP offloading")
  # Look for OFFLOAD_TARGET_NAMES in `g++ -v` to get otpions
  # CUDA doesn't like extra protections that gcc adds by default
  set(OPENMP_OFFLOADING_CFLAGS  "-foffload=nvptx-none=\"-fcf-protection=none -fno-stack-protector\"" CACHE STRING "Compiler arguments for OpenMP offloading")
  set(OPENMP_OFFLOADING_LDFLAGS "-foffload=nvptx-none=\"-fcf-protection=none -fno-stack-protector\"" CACHE STRING "Linker arguments for OpenMP offloading")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(OPENMP_OFFLOADING_CFLAGS
      "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70"
      CACHE STRING "Compiler arguments for OpenMP offloading")
  set(OPENMP_OFFLOADING_LDFLAGS
      "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70;-lomptarget;-v"
      CACHE STRING "Linker arguments for OpenMP offloading")
endif ()

# Doesn't mean that it actually offloads, just that it compiles and links.
set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_CFLAGS})
set(CMAKE_REQUIRED_INCLUDES ${OpenMP_CXX_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LINK_OPTIONS ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_LDFLAGS})
set(CMAKE_REQUIRED_LIBRARIES ${OpenMP_CXX_LIBRARIES})
check_cxx_source_compiles(
  "
        int main(void) {
        // int a ;
          #pragma omp target // teams distribute parallel for map(from:a)
          {
             //     for (int i = 0; i < 128; ++i) {
              //          a = 0;
               //   }
          }
          return 0;
        }
    "
  HAVE_PRAGMA_OMP_TARGET)

if (HAVE_PRAGMA_OMP_TARGET)
  set(OpenMPOffload_FOUND TRUE)
  message(STATUS "Found OpenMP Offloading: ${OPENMP_OFFLOADING_CFLAGS}")

  add_library(OpenMP::OpenMP_Offload_CXX INTERFACE IMPORTED)
  target_link_libraries(OpenMP::OpenMP_Offload_CXX INTERFACE OpenMP::OpenMP_CXX)
  set_property(TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OPENMP_OFFLOADING_CFLAGS})
  set_property(
    TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_FLAGS} ${OPENMP_OFFLOADING_LDFLAGS}
  )# FIXME: OpenMP_CXX_FLAGS (-fopenmp) already be added by target_link_libraries
endif ()
