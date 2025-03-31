cmake_minimum_required( VERSION 3.14 )
project (myslam)

# cpp
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
endif()

# Paths
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

############### dependencies ######################
# Eigen (header-only)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus (header-only)
find_package(Sophus REQUIRED)
include_directories(${SOPHUS_INCLUDE_DIR})

# OpenCV
find_package(OpenCV 3.1 REQUIRED PATHS "/Users/bowen/Source/3rd_party/build_opencv")
include_directories(${OpenCV_INCLUDE_DIRS})

# G2O
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

# csparse
find_package(CSPARSE REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# pangolin
find_package(Pangolin REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS timer)

# libtorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIR})

set(THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${CSPARSE_LIBRARY}
    ${Pangolin_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${TORCH_LIBRARIES}
)

############### source and test ######################
add_subdirectory(src)
target_include_directories(myslam 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/include/myslam/private
)

add_subdirectory( app )
# add_subdirectory( gtest )