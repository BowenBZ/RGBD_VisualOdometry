enable_testing()

include(GoogleTest)

function(add_gtest test_name)
  add_executable(${test_name} "${PROJECT_SOURCE_DIR}/test/${test_name}.cpp")
  target_link_libraries(${test_name} gtest gtest_main myslam ${THIRD_PARTY_LIBS})
  gtest_discover_tests(${test_name})
endfunction()

add_gtest(basis_test)
add_gtest(frontend_test)
