# build executables
add_executable(test-classification test_classification.cpp)
add_executable(test-clustering test_clustering.cpp)
add_executable(test-data test_data.cpp)
add_executable(test-matrix test_matrix.cpp)

# link mlearn library to executables
target_link_libraries(test-classification LINK_PUBLIC mlearn)
target_link_libraries(test-clustering LINK_PUBLIC mlearn)
target_link_libraries(test-data LINK_PUBLIC mlearn)
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
		test-data
		test-matrix
	RUNTIME DESTINATION bin
	COMPONENT dev
)
