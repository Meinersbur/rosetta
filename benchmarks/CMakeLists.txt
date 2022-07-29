









# internal
# TODO: basename is not required
function (add_benchmark_ppm basename ppm ppm_shortname _library)
  cmake_parse_arguments(_arg "" "" "SOURCES" ${ARGN})

  file(RELATIVE_PATH _relpath  "${CMAKE_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}")
  string(REPLACE "/" "." _relpath_dots "${_relpath}")
  set(benchname "${_relpath_dots}") 
  set(basename "${_relpath_dots}")

  set(_genconfig_target "genconfig-${benchname}")
  set(_benchprop_filename "${benchname}.$<CONFIG>.benchprop.cxx")
  # TODO: Simpler if using configure_file?
  if (NOT TARGET "${_genconfig_target}")
    add_custom_command(
      OUTPUT "${_benchprop_filename}"
      DEPENDS "${_problemsizefile}" "${GENCONFIG_EXECUTABLE}"
      COMMAND Python::Interpreter "${GENCONFIG_EXECUTABLE}" 
        "--benchname=${benchname}"
        "--problemsizefile=${_problemsizefile}"
        "--output=${_benchprop_filename}"
        "--resultsdir=${ROSETTA_RESULTS_DIR}"
        "--compiler=${CMAKE_CXX_COMPILER}" # FIXME: Depends on PPM
        "--compiler-flags=${CMAKE_CXX_FLAGS}" # FIXME: Not complete and depends on configuration and PPM
        "--configname=${ROSETTA_CONFIGNAME}"
        "--config=$<CONFIG>"
      COMMENT "Generate ${benchname} $<CONFIG> properties file ..."
    )
    add_custom_target("${_genconfig_target}"
      DEPENDS "${_benchprop_filename}"
    )
  endif ()

  foreach(_buildtype IN LISTS _build_types)
  endforeach()
   


  set(_bench_target "${basename}.${ppm_shortname}")
  set(_sources ${_arg_SOURCES} "${_benchprop_filename}")

  string(REPLACE "-" "_" ppm_mixed "${ppm}")

  add_executable("${_bench_target}" ${_sources})
  set_target_properties("${_bench_target}" 
    PROPERTIES 
      FOLDER "Benchmarks"
      ROSETTA_BASENAME "${basename}"
    )
  target_link_libraries("${_bench_target}" PRIVATE "${_library}")
  if (ROSETTA_REALTYPE)
    # TODO: Support compiling mutiple real types at the same time?
    target_compile_definitions(${_bench_target} PRIVATE "ROSETTA_REALTYPE=${ROSETTA_REALTYPE}")
  endif ()


  get_property(_reffile TARGET "${basename}.serial" PROPERTY REFERENCE_FILE)


  set_property(TARGET ${_bench_target} PROPERTY REFERENCE_FILE "${_reffile}")
  #set_property(TARGET ${_verify_target} PROPERTY REFERENCE_FILE "${_reffile}")
  set_property(TARGET ${_bench_target} PROPERTY VERIFY_TARGET "${_verify_target}")

  append_global_property("benchmarks_${ppm_mixed}" "${_bench_target}")
  #append_global_property("verifications_${ppm_mixed}" "${_verify_target}")


  #add_custom_target("${basename}.${ppm}")
  #add_dependencies("${basename}.${ppm}" "${_bench_target}" "${_verify_target}")

  add_dependencies("build-${ppm}" "${_bench_target}")
  #add_dependencies("build-${ppm}-verify" "${_verify_target}")
endfunction ()




function (make_reference_output basename ppm ppm_shortname _library)
  set(_reference_dir "${CMAKE_PROJECT_BINARY_DIR}/reference_outputs")
  if (CMAKE_BUILD_TYPE)
    set(_reference_filename "${basename}.reference")
  else ()
    set(_reference_filename "${basename}.$<CONFIG>.reference")
  endif ()
  set(_reffile "${_reference_dir}/${_reference_filename}")

  # TODO: different reference file for each problemsize? Hardcode problemsize for verification?
  add_custom_command(
    OUTPUT "${_reference_filename}"
    DEPENDS "${_bench_target}" "${RUNNER_EXECUTABLE}"
    COMMAND Python::Interpreter "${RUNNER_EXECUTABLE}" "--problemsizefile=${_problemsizefile}" "--gen-reference" "$<TARGET_FILE:${_bench_target}>" "${_reffile}"
    COMMENT "Generate ${basename} reference output (using serial output) ..."
  )
  add_custom_target("reference-${basename}"
    DEPENDS "${_reference_filename}"
  )
  append_global_property(references "reference-${basename}")
  set_property(TARGET ${_verify_target}    
    PROPERTY REFERENCE_FILE "${_reference}"
  )
  append_global_property(references_serial "${_reference}")
endfunction()



function (add_benchmark_serial basename)
  if (NOT ROSETTA_PPM_SERIAL)
    return ()
  endif ()
  add_benchmark_ppm("${basename}" "serial" "serial" "rosetta-serial" ${ARGN} SOURCES "${basename}.cxx")
  #make_reference_output("${basename}" "serial" "serial" "rosetta-serial")
endfunction ()


function (add_benchmark_cuda basename)
  if (NOT ROSETTA_PPM_CUDA)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "cuda" "cuda" "rosetta-cuda" ${ARGN} SOURCES "${basename}.cu")
endfunction ()


function (add_benchmark_openmp_parallel basename)
  if (NOT ROSETTA_PPM_OPENMP_PARALLEL)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-parallel" "omp_parallel" "rosetta-openmp-parallel" ${ARGN} SOURCES "${basename}.omp_parallel.cxx")
endfunction ()


function (add_benchmark_openmp_task basename)
  if (NOT ROSETTA_PPM_OPENMP_TASK)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-task" "omp_task" "rosetta-openmp-task" ${ARGN} SOURCES "${basename}.omp_task.cxx")
endfunction ()


function (add_benchmark_openmp_target basename)
  if (NOT ROSETTA_PPM_OPENMP_TARGET)
      return ()
  endif ()
  add_benchmark_ppm("${basename}" "openmp-target" "omp_target" "rosetta-openmp-target" ${ARGN} SOURCES "${basename}.omp_target.cxx")
endfunction ()







function (add_benchmark basename)
  #cmake_parse_arguments(ARG "" "NAME" "SERIAL;CUDA;OMP_PARALLEL;OMP_TASK;OMP_TARGET" ${ARGN})

  set(_source_serial "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cpp")
  if (EXISTS "${_source_serial}")
    add_benchmark_serial("${basename}" SOURCES ${_source_serial})
  endif ()

  set(_source_cuda "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cu")
  if (EXISTS "${_source_cuda}")
    add_benchmark_cuda("${basename}" SOURCES ${_source_cuda})
  endif ()

  set(_source_openmp_parallel "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_parallel.cpp")
  if (EXISTS "${_source_cuda}")
    add_benchmark_openmp_parallel("${basename}" SOURCES ${_source_openmp_parallel})
  endif ()

  set(_source_openmp_task "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_task.cpp")
  if (EXISTS "${_source_openmp_task}")
    add_benchmark_openmp_task("${basename}" SOURCES ${_source_openmp_task})
  endif ()

  set(_source_openmp_target "${CMAKE_CURRENT_SOURCE_DIR}/${basename}.openmp_target.cpp")
  if (EXISTS "${_source_openmp_target}")
    add_benchmark_openmp_target("${basename}" SOURCES ${_source_openmp_target})
  endif ()
endfunction ()





add_subdirectory(idioms)
#add_subdirectory(features)
#add_subdirectory(synth)
add_subdirectory(suites)
#add_subdirectory(apps)
