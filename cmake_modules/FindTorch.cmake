set(TORCH_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../dependency/libtorch")
set(TORCH_LIB_DIR "${TORCH_BASE_DIR}/lib")

# Look for torch
find_path(TORCH_SCRIPT
  NAMES torch/script.h
  PATHS
  "${TORCH_BASE_DIR}/include"
)

find_path(TORCH_NN
  NAMES torch/nn/functional.h
  PATHS
  "${TORCH_BASE_DIR}/include/torch/csrc/api/include"
)

FIND_LIBRARY(TORCH_CORE
  NAMES torch
  PATHS
  ${TORCH_LIB_DIR}
)

FIND_LIBRARY(TORCH_C10 
  NAMES c10
  PATHS
  ${TORCH_LIB_DIR}
)

FIND_LIBRARY(TORCH_CPU
  NAMES torch_cpu
  PATHS
  ${TORCH_LIB_DIR}
)

if(TORCH_SCRIPT AND TORCH_NN AND TORCH_CORE AND TORCH_C10 AND TORCH_CPU)
  set(TORCH_INCLUDE_DIR
    ${TORCH_SCRIPT}
    ${TORCH_NN})
  set(TORCH_LIBRARIES
    ${TORCH_CORE}
    ${TORCH_C10}
    ${TORCH_CPU})
  message(STATUS "Found Torch in: ${TORCH_INCLUDE_DIR}, ${TORCH_LIBRARIES}")
else()
  message(WARNING "Torch not found.")
endif()