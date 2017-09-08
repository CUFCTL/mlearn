# Create a library called "mlearn" which includes the source files.
# The extension is already found. Any number of sources could be listed here.
include_directories(${CMAKE_SOURCE_DIR}/src)

# external libraries
set(OPENBLASDIR "$ENV{HOME}/software/OpenBLAS-0.2.19")
add_library(openblas SHARED IMPORTED)
set_target_properties(openblas PROPERTIES
	IMPORTED_LOCATION "${OPENBLASDIR}/lib/libopenblas.so"
	INTERFACE_INCLUDE_DIRECTORIES "${OPENBLASDIR}/include"
)

set(MAGMADIR "$ENV{HOME}/software/magma-2.2.0")
add_library(magma SHARED IMPORTED)
set_target_properties(magma PROPERTIES
	IMPORTED_LOCATION "${MAGMADIR}/lib/libmagma.so"
	INTERFACE_INCLUDE_DIRECTORIES "${MAGMADIR}/include"
)
include_directories("${MAGMADIR}/include")

# build mlearn library
file(GLOB_RECURSE mlearn_src ${CMAKE_SOURCE_DIR}/src/*.cpp)

set_source_files_properties(${mlearn_src} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_library(mlearn SHARED ${mlearn_src})
cuda_add_cublas_to_target(mlearn)
target_link_libraries(mlearn openblas magma)

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
		${CMAKE_SOURCE_DIR}/src/clustering
		${CMAKE_SOURCE_DIR}/src/criterion
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
