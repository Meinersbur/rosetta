

add_library(rosetta
  rosetta.cpp
  )
target_include_directories(rosetta PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(rosetta PUBLIC benchmark::benchmark)

if (ROSETTA_NO_EXCEPTIONS)
  # TODO: Introspection
  target_compile_options(rosetta PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>) 
endif ()


add_subdirectory(idioms)
add_subdirectory(synth)
#add_subdirectory(suites)
#add_subdirectory(apps)

set(OPENMP_OFFLOADING_CFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70"
  CACHE STRING "Compiler arguments for OpenMP offloading")
set(OPENMP_OFFLOADING_LDFLAGS "-fopenmp-targets=nvptx64-nvidia-cuda;-Xopenmp-target;-march=sm_70;-lomptarget"
  CACHE STRING "Linker arguments for OpenMP offloading")