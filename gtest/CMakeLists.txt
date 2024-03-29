set( CMAKE_CXX_COMPILER "/usr/bin/c++" )

# gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  basis_test
  ${PROJECT_SOURCE_DIR}/gtest/basis_test.cpp
)
target_link_libraries(
  basis_test
  gtest_main
  ${THIRD_PARTY_LIBS}
)

include(GoogleTest)
gtest_discover_tests(basis_test)