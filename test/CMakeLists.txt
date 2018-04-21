cmake_minimum_required(VERSION 3.3)
project(MLEARN)

# build parameters
if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# compiler settings
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-unused-result")
set(CMAKE_EXE_LINKER_FLAGS "-L$ENV{INSTALL_PREFIX}/lib")

include_directories($ENV{CUDADIR}/include)
include_directories($ENV{INSTALL_PREFIX}/include)

# build executables
add_executable(test-classification test_classification.cpp)
add_executable(test-clustering test_clustering.cpp)
add_executable(test-matrix test_matrix.cpp)

# link mlearn library to executables
target_link_libraries(test-classification LINK_PUBLIC mlearn)
target_link_libraries(test-clustering LINK_PUBLIC mlearn)
target_link_libraries(test-matrix LINK_PUBLIC mlearn)

# install tests
install(
	DIRECTORY data
	DESTINATION test
	COMPONENT dev
)

install(
	TARGETS
		test-classification
		test-clustering
		test-matrix
	RUNTIME DESTINATION bin
	COMPONENT dev
)
