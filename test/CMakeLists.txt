# build executables
add_executable(test-genome test_genome.cpp)
add_executable(test-image test_image.cpp)
add_executable(test-matrix test_matrix.cpp)

# link mlearn library to executables
target_link_libraries(test-genome LINK_PUBLIC mlearn)
target_link_libraries(test-image LINK_PUBLIC mlearn)
target_link_libraries(test-matrix LINK_PUBLIC mlearn)

# install tests
install(
	TARGETS test-genome test-image test-matrix
	RUNTIME DESTINATION bin
	COMPONENT runtime
)