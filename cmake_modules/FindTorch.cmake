# Look for torch
find_path(TORCH_SCRIPT
  NAMES torch/script.h
  PATHS
  /Users/bowen/Source/3rd_party/libtorch/include
)

find_path(TORCH_NN
  NAMES torch/nn/functional.h
  PATHS
  /Users/bowen/Source/3rd_party/libtorch/include/torch/csrc/api/include
)

FIND_LIBRARY(TORCH_CORE
  NAMES torch
  PATHS
  /Users/bowen/Source/3rd_party/libtorch/lib
)

FIND_LIBRARY(TORCH_C10 
  NAMES c10
  PATHS
  /Users/bowen/Source/3rd_party/libtorch/lib
)

FIND_LIBRARY(TORCH_CPU
  NAMES torch_cpu
  PATHS
  /Users/bowen/Source/3rd_party/libtorch/lib
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