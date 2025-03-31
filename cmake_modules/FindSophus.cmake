# Look for Sophus
find_path(SOPHUS_INCLUDE_DIR
  NAMES sophus/so3.hpp
  PATHS
  /Users/bowen/Source/3rd_party/Sophus
)

if(SOPHUS_INCLUDE_DIR)
  message(STATUS "Found Sophus in: ${SOPHUS_INCLUDE_DIR}")
else()
  message(WARNING "Sophus not found.")
endif()