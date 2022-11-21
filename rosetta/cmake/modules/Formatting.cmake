# TODO: Introspect availability of clang-format, autopip8/flake8/black, cmake-format TODO: Should be called multiple
# times with different .clang-format configurations for benchmarks/ and rosetta/
function (add_format_target)
  cmake_parse_arguments(_arg "" "" "CLANG_RGLOB;PY_RGLOB;CMAKE_RGLOB" ${ARGN})

  set(check_format_depends)
  set(update_format_depends)
  set(i 0)

  file(GLOB_RECURSE format_files ${_arg_CLANG_RGLOB})
  foreach (file IN LISTS format_files)
    set(_stamp "stamps/check-format-stamp${i}")
    list(APPEND check_format_depends "${_stamp}")
    set_source_files_properties("{_stamp}" PROPERTIES SYMBOLIC "true")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${CMAKE_SOURCE_DIR}/.clang-format" "${file}"
      COMMAND clang-format "${file}" --dry-run --output-replacements-xml --color=1 -Werror # | diff -u --color=always
                                                                                           # ${file} -
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

  file(GLOB_RECURSE format_cmake_files CMakeLists.txt ${_arg_CMAKE_RGLOB})
  foreach (file IN LISTS format_cmake_files)
    set(_stamp "stamps/check-cmake-format-stamp${i}")
    list(APPEND check_format_depends "${_stamp}")
    set_source_files_properties("{_stamp}" PROPERTIES SYMBOLIC "true")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND cmake-format --check --line-width 120 --separate-ctrl-name-with-space true --line-ending unix "${file}"
      VERBATIM
      COMMENT "Check format of ${file}")

    set(_stamp "stamps/update-cmake-format-stamp${i}")
    list(APPEND update_format_depends "${_stamp}")
    add_custom_command(
      OUTPUT "${_stamp}"
      DEPENDS "${file}"
      COMMAND cmake-format -i --line-width 120 --separate-ctrl-name-with-space true --line-ending unix "${file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
      COMMAND "${CMAKE_COMMAND}" -E echo "${file}" "${_stamp}"
      VERBATIM
      COMMENT "Update format of ${file}")
    # message("file: ${file} ${_stamp}")

    math(EXPR i ${i}+1)
  endforeach ()

  # message("update_format_depends: ${update_format_depends}")
  add_custom_target(check-format DEPENDS ${check_format_depends})
  add_custom_target(update-format DEPENDS ${update_format_depends})
  set_target_properties("check-format" PROPERTIES FOLDER "Maintanance")
  set_target_properties("update-format" PROPERTIES FOLDER "Maintanance")
endfunction ()
