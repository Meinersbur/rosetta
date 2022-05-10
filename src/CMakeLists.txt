if (HAVE_CXX_RESTRICT)
  set (KEYWORD_RESTRICT "restrict")
elif (HAVE_CXX___RESTRICT)
  set (KEYWORD_RESTRICT "__restrict")
else ()
  set (KEYWORD_RESTRICT "")
endif ()


function (append_global_property propname)
  get_property(propval GLOBAL PROPERTY ${propname})
  message("${propname} ${propval}")
  list(APPEND propval ${ARGN})
  set_property(GLOBAL PROPERTY ${propname} "${propval}")
endfunction ()


function (add_test_serial name)
  if (NOT XCELLENT_ENABLE_SERIAL)
    return()
  endif ()

  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "")
  set(exe "${name}.serial")
  set(src "${name}.cpp")

  add_executable(${exe} "${src}")
  target_link_libraries(${exe} PRIVATE benchmark::benchmark)
  #target_compile_options(${exe} PUBLIC -fno-exceptions)
  add_dependencies(gbench.serial ${exe})

  append_global_property(GBENCHS_SERIAL_EXE "$<TARGET_FILE:${exe}>")
  append_global_property(GBENCHS_EXE "$<TARGET_FILE:${exe}>")
endfunction ()


function (add_test_cuda name)
  if (NOT XCELLENT_ENABLE_CUDA)
    return()
  endif ()

  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "")
  set(exe "${name}.set")
  set(src "${name}.cu")

  add_executable(${exe} "${src}")
  target_link_libraries(${exe} PRIVATE benchmark::benchmark)
  add_dependencies(gbench.cuda ${exe})

  append_global_property(GBENCHS_CUDA_EXE "$<TARGET_FILE:${exe}>")
  append_global_property(GBENCHS_EXE "$<TARGET_FILE:${exe}>")
endfunction ()



add_subdirectory(idioms)
add_subdirectory(synth)
#add_subdirectory(suites)
#add_subdirectory(apps)
