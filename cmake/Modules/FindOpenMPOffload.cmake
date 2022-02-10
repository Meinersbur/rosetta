
# find module
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules

# Inputs:
#   OpenMPOffload_FIND_VERSION
#   OpenMPOffload_FIND_QUIETLY
#   OpenMPOffload_FIND_REQUIRED

# Outputs:
#  OpenMPOffload_FOUND
#  OpenMPOffload_DEFINITIONS
#  OpenMPOffload import target

if (NOT DEFINED OpenMP_FOUND)
  if (OpenMPOffload_FIND_REQUIRED)
    find_package(OpenMP REQUIRED)
  else ()
    find_package(OpenMP OPTIONAL)
  endif ()
endif ()

add_library(OpenMP::OpenMP_Offload_CXX INTERFACE IMPORTED)
set_property(TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OPENMP_OFFLOADING_CFLAGS})
set_property(TARGET OpenMP::OpenMP_Offload_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OPENMP_OFFLOADING_LDFLAGS})

set(OpenMPOffload_FOUND TRUE)
