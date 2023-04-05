# TODO: Introspect availability of clang-format, autopip8/flake8/black, cmake-format 
# TODO: Should be called multiple times with different .clang-format configurations for benchmarks/ and rosetta/
# TODO: Ensure that the .git directory is not globbed recursively, very slow in WSL2
function (add_format_target)
  cmake_parse_arguments(_arg "" "" "CLANG_RGLOB;PY_RGLOB;CMAKE;CMAKE_RGLOB" ${ARGN})

  message("Searching for files that can be automatically formatted...")

  set(check_format_depends)
  set(update_format_depends)
  set(i 0)

  message("Globbing ${_arg_CLANG_RGLOB} ...")
  file(GLOB_RECURSE format_files ${_arg_CLANG_RGLOB})
  foreach (file IN LISTS format_files)
    set(_stamp "stamps/check-format-stamp${i}")
    list(APPEND check_format_depends "${_stamp}")
    set_source_files_properties("{_stamp}" PROPERTIES SYMBOLIC "true")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${CMAKE_SOURCE_DIR}/.clang-format" "${file}"
      COMMAND clang-format "${file}" --dry-run --output-replacements-xml --color=1 -Werror
      VERBATIM
      COMMENT "Check format of ${file}")

    set(_stamp "stamps/update-format-stamp${i}")
    list(APPEND update_format_depends "${_stamp}")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${CMAKE_SOURCE_DIR}/.clang-format" "${file}"
      COMMAND clang-format -i "${file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
      VERBATIM
      COMMENT "Update format of ${file}")

    math(EXPR i ${i}+1)
  endforeach ()


  message("Globbing ${_arg_PY_RGLOB} ...")
  file(GLOB_RECURSE format_py_files ${_arg_PY_RGLOB})
  foreach (file IN LISTS format_py_files)
    set(_stamp "stamps/check-py-format-stamp${i}")
    list(APPEND check_format_depends "${_stamp}")
    set_source_files_properties("{_stamp}" PROPERTIES SYMBOLIC "true")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND autopep8 --max-line-length 120 -d -a "${file}"
      VERBATIM
      COMMENT "Check format of ${file}")

    set(_stamp "stamps/update-py-format-stamp${i}")
    list(APPEND update_format_depends "${_stamp}")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND autopep8 --max-line-length 120 -i -a "${file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
      VERBATIM
      COMMENT "Update format of ${file}")

    math(EXPR i ${i}+1)
  endforeach ()


  message("Globbing ${_arg_CMAKE_RGLOB} ...")
  file(GLOB_RECURSE format_cmake_files ${_arg_CMAKE_RGLOB})
  list(APPEND format_cmake_files ${_arg_CMAKE})
  foreach (file IN LISTS format_cmake_files)
    set(_stamp "stamps/check-cmake-format-stamp${i}")
    list(APPEND check_format_depends "${_stamp}")
    set_source_files_properties("{_stamp}" PROPERTIES SYMBOLIC "true")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND cmake-format --check --line-width 120 --separate-ctrl-name-with-space true --enable-markup false --max-statement-spacing 2 --output-encoding utf-8 --line-ending unix "${file}"
      VERBATIM
      COMMENT "Check format of ${file}")

    set(_stamp "stamps/update-cmake-format-stamp${i}")
    list(APPEND update_format_depends "${_stamp}")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND cmake-format -i --line-width 120 --separate-ctrl-name-with-space true --enable-markup false --max-statement-spacing 2 --output-encoding utf-8 --line-ending unix "${file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
      VERBATIM
      COMMENT "Update format of ${file}")

    math(EXPR i ${i}+1)
  endforeach ()

  add_custom_target(check-format DEPENDS ${check_format_depends})
  add_custom_target(update-format DEPENDS ${update_format_depends})
  set_target_properties("check-format" PROPERTIES FOLDER "Maintanance")
  set_target_properties("update-format" PROPERTIES FOLDER "Maintanance")


  message("... done searching")
endfunction ()
