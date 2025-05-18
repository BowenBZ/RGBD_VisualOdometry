set(G2O_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../dependency/g2o")
set(G2O_LIB_DIR "${G2O_BASE_DIR}/lib")

# Find the header files
find_path(G2O_INCLUDE_DIR 
  NAME g2o/core/base_vertex.h
  PATHS
  ${G2O_BASE_DIR}
)

find_path(G2O_CONFIG_DIR 
  NAME g2o/config.h
  PATHS
  "${G2O_BASE_DIR}/build"
)

# Macro to unify finding both the debug and release versions of the
# libraries; this is adapted from the OpenSceneGraph FIND_LIBRARY
# macro.

MACRO(FIND_G2O_LIBRARY MYLIBRARY MYLIBRARYNAME)

  FIND_LIBRARY("${MYLIBRARY}_DEBUG"
    NAMES "g2o_${MYLIBRARYNAME}_d"
    PATHS
    ${G2O_LIB_DIR}
  )

  FIND_LIBRARY(${MYLIBRARY}
    NAMES "g2o_${MYLIBRARYNAME}"
    PATHS
    ${G2O_LIB_DIR}
   )
  
  IF(NOT ${MYLIBRARY}_DEBUG)
    IF(MYLIBRARY)
      SET(${MYLIBRARY}_DEBUG ${MYLIBRARY})
    ENDIF(MYLIBRARY)
  ENDIF( NOT ${MYLIBRARY}_DEBUG)
  
ENDMACRO(FIND_G2O_LIBRARY LIBRARY LIBRARYNAME)

# Find the core elements
FIND_G2O_LIBRARY(G2O_STUFF_LIBRARY stuff)
FIND_G2O_LIBRARY(G2O_CORE_LIBRARY core)

# Find the pluggable solvers
FIND_G2O_LIBRARY(G2O_SOLVER_CSPARSE_EXTENSION csparse_extension)

if(G2O_INCLUDE_DIR AND G2O_CONFIG_DIR AND G2O_STUFF_LIBRARY AND G2O_CORE_LIBRARY AND G2O_SOLVER_CSPARSE_EXTENSION)
  set(G2O_INCLUDE_DIR 
    ${G2O_INCLUDE_DIR} 
    ${G2O_CONFIG_DIR})
  set(G2O_LIBS 
    ${G2O_STUFF_LIBRARY} 
    ${G2O_CORE_LIBRARY} 
    ${G2O_SOLVER_CSPARSE_EXTENSION} 
  )
  message(STATUS "Found g2o in: ${G2O_INCLUDE_DIR}, ${G2O_LIBS}")
else()
  message(WARNING "g2o not found.")
endif()