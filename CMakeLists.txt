cmake_minimum_required(VERSION 3.3)
project(MLEARN)

# build parameters
if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# compiler settings
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDADIR})
find_package(CUDA REQUIRED)

include(FindCUDA)

set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-unused-result")

# recurse into subdirectories
add_subdirectory(src)

# install additional files
install(
	FILES "README.md" "LICENSE"
	DESTINATION share/mlearn
	COMPONENT dev
)
