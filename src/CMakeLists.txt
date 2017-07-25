# Create a library called "mlearn" which includes the source files.
# The extension is already found. Any number of sources could be listed here.
include_directories(${CMAKE_SOURCE_DIR}/src)

# external libraries
add_library(openblas SHARED IMPORTED)
set_target_properties(openblas PROPERTIES
	IMPORTED_LOCATION "$ENV{HOME}/software/OpenBLAS-0.2.19/lib/libopenblas.so"
	INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/software/OpenBLAS-0.2.19/include"
)

# build mlearn library
file(GLOB_RECURSE mlearn_src ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(mlearn SHARED ${mlearn_src})

target_link_libraries(mlearn openblas)

# install libmlearn.so
install(
	TARGETS mlearn
	LIBRARY DESTINATION lib
	COMPONENT runtime
)

# install mlearn.h
install(
	FILES ${CMAKE_SOURCE_DIR}/src/mlearn.h
	DESTINATION include
	COMPONENT dev
)

# install header files
install(
	DIRECTORY
		${CMAKE_SOURCE_DIR}/src/classifier
		${CMAKE_SOURCE_DIR}/src/data
		${CMAKE_SOURCE_DIR}/src/feature
		${CMAKE_SOURCE_DIR}/src/math
		${CMAKE_SOURCE_DIR}/src/model
		${CMAKE_SOURCE_DIR}/src/util
	DESTINATION include
	COMPONENT dev
	FILES_MATCHING
		PATTERN "*.h"
		PATTERN "mlearn.h" EXCLUDE
)

# Make sure the compiler can find include files for mlearn
# when other libraries or executables link to mlearn
target_include_directories(mlearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
