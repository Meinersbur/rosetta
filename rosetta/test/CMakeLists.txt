# Python Unittest ##############################################################

execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/list_tests.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE STR_TESTS
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

separate_arguments(TEST_LIST UNIX_COMMAND ${STR_TESTS})

foreach (ATEST ${TEST_LIST})
  add_test(
    NAME "${ATEST}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND "${Python_EXECUTABLE}" "-m" "unittest" "-v" "${ATEST}")
endforeach (ATEST)

# Catch2Powered C++ tests ######################################################

if (TARGET rosetta-serial)
  add_executable(cpptests test_stat.cpp)
  target_link_libraries(cpptests PRIVATE Catch2::Catch2WithMain)
  target_link_libraries(cpptests PRIVATE rosetta-serial)
  target_include_directories(cpptests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
  set_target_properties(cpptests PROPERTIES FOLDER "Self-Testing")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(cpptests)
endif ()
