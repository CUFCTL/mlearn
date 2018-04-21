# Create a library called "mlearn" which includes the source files.
# The extension is already found. Any number of sources could be listed here.
include_directories(${CMAKE_SOURCE_DIR}/src)

# build mlearn library
file(GLOB_RECURSE mlearn_src ${CMAKE_SOURCE_DIR}/src/*.cpp)

set_source_files_properties(${mlearn_src} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_library(mlearn SHARED ${mlearn_src})
target_link_libraries(mlearn blas lapacke -L$ENV{CUDADIR}/lib64 cudart cublas cusolver)

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
		${CMAKE_SOURCE_DIR}/src/mlearn
	DESTINATION include
	COMPONENT dev
	FILES_MATCHING
		PATTERN "*.h"
)

# Make sure the compiler can find include files for mlearn
# when other libraries or executables link to mlearn
target_include_directories(mlearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
