# CMakeLists files in this project can
# refer to the root source directory of the project as ${MLEARN_SOURCE_DIR} and
# to the root binary directory of the project as ${MLEARN_BINARY_DIR}.
cmake_minimum_required(VERSION 3.3)
project(MLEARN)

# build parameters
if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
	set(CMAKE_BUILD_TYPE "Release")
endif ()

if ( NOT DEFINED GPU )
	set(GPU "1")
endif ()

# compiler settings
if ( GPU STREQUAL "1" )
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
	find_package(CUDA REQUIRED)

	include(FindCUDA)

	set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif ()

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-unused-result")

# recurse into subdirectories
add_subdirectory(src)
add_subdirectory(test)

# install additional files
install(
	FILES "README.md" "LICENSE"
	DESTINATION share/mlearn
	COMPONENT dev
)
