# Look for Sophus
find_path(SOPHUS_INCLUDE_DIR
  NAMES sophus/so3.hpp
  PATHS
  "${CMAKE_CURRENT_LIST_DIR}/../dependency/Sophus"
)

if(SOPHUS_INCLUDE_DIR)
  message(STATUS "Found Sophus in: ${SOPHUS_INCLUDE_DIR}")
else()
  message(WARNING "Sophus not found.")
endif()