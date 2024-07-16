cmake_minimum_required( VERSION 3.14 )
project ( myslam )

set( CMAKE_CXX_COMPILER "g++" )

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set( CMAKE_BUILD_TYPE "Debug" )
    set( CMAKE_CXX_FLAGS "-std=c++17")
else()
    set( CMAKE_BUILD_TYPE "Release" )
    set( CMAKE_CXX_FLAGS "-std=c++17 -march=native -O3" )
endif()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
include_directories( "/usr/local/include/eigen3" )

# Sophus
include_directories( "/Users/bowen/Source/3rd_party/Sophus" )

# OpenCV
find_package( OpenCV 3.1 REQUIRED PATHS "/Users/bowen/Source/3rd_party/build_opencv")
include_directories( ${OpenCV_INCLUDE_DIRS} )

# G2O
set(G2O_LIBRARY_DIR "/Users/bowen/Source/3rd_party/g2o/lib")
find_library(G2O_CORE_LIBRARY g2o_core PATHS ${G2O_LIBRARY_DIR})
find_library(G2O_STUFF_LIBRARY g2o_stuff PATHS ${G2O_LIBRARY_DIR})
find_library(G2O_CSPARSE_EXTENSION_LIBRARY g2o_csparse_extension PATHS ${G2O_LIBRARY_DIR})
include_directories("/Users/bowen/Source/3rd_party/g2o")
include_directories("/Users/bowen/Source/3rd_party/build_g2o")

# Still needs following
# export DYLD_LIBRARY_PATH=/Users/bowen/Source/3rd_party/g2o/lib:$DYLD_LIBRARY_PATH

# csparse
set(CSPARSE_INCLUDE_DIR "/usr/local/include/suitesparse")
find_package( CSPARSE REQUIRED )
include_directories(${CSPARSE_INCLUDE_DIR})

# pangolin
find_package( Pangolin REQUIRED PATHS "/Users/bowen/Source/3rd_party/build_Pangolin" )

# Boost
find_package(Boost REQUIRED COMPONENTS timer)

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_CSPARSE_EXTENSION_LIBRARY}
    ${CSPARSE_LIBRARY}
    ${Pangolin_LIBRARIES} 
    ${Boost_LIBRARIES}
)
############### source and test ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( app )
# add_subdirectory( gtest )