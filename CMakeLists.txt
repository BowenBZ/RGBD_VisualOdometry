cmake_minimum_required( VERSION 3.14 )
project ( myslam )

set( CMAKE_CXX_COMPILER "g++" )

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set( CMAKE_BUILD_TYPE "Debug" )
    set( CMAKE_CXX_FLAGS "-std=c++11")
else()
    set( CMAKE_BUILD_TYPE "Release" )
    set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )
endif()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )

# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

# csparse
find_package( CSPARSE REQUIRED )
include_directories(${CSPARSE_INCLUDE_DIR})

# pangolin
find_package( Pangolin REQUIRED )
include_directories(${Pangolin_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS 
    boost_timer
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES} GL GLU GLEW glut
    g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
    ${CSPARSE_LIBRARY}
)
############### source and test ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( app )
add_subdirectory( gtest )