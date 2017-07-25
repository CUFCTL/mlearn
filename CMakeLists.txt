# CMakeLists files in this project can
# refer to the root source directory of the project as ${MLEARN_SOURCE_DIR} and
# to the root binary directory of the project as ${MLEARN_BINARY_DIR}.
cmake_minimum_required(VERSION 2.8.11)
project(MLEARN)

# build type
if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# compiler settings
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-unused-result")

# echo configuration settings
message("CMAKE_BUILD_TYPE:        " ${CMAKE_BUILD_TYPE})
message("CMAKE_CXX_FLAGS:         " ${CMAKE_CXX_FLAGS})
message("CMAKE_CXX_FLAGS_DEBUG:   " ${CMAKE_CXX_FLAGS_DEBUG})
message("CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message("CMAKE_INSTALL_PREFIX:    " ${CMAKE_INSTALL_PREFIX})

add_subdirectory(src)
add_subdirectory(test)

install(
	FILES "README.md" "LICENSE"
	DESTINATION share/mlearn
	COMPONENT dev
)
