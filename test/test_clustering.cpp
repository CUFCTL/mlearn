/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <iostream>
#include <mlearn.h>

using namespace ML;

int main(int argc, char **argv)
{
	if ( argc < 2 ) {
		std::cerr << "usage: ./test-clustering [clusters...]\n";
		exit(1);
	}

	const char *filename = "test/data/iris.train";
	std::vector<int> clusters;

	for ( int i = 1; i < argc; i++ ) {
		clusters.push_back(atoi(argv[i]));
	}

	// load input dataset
	Dataset input_data(nullptr, filename);

	// create clustering model
	ClusteringLayer *clustering = new KMeansLayer();
	ClusteringModel model(clusters, clustering);

	// perform clustering on input data
	std::vector<int> Y_pred = model.run(input_data);

	// print clustering results
	log(LL_INFO, "Results");

	for ( size_t i = 0; i < input_data.entries().size(); i++ ) {
		int y_pred = Y_pred[i];
		const DataEntry& entry = input_data.entries()[i];

		log(LL_INFO, "%-4s (%s) -> %d",
			entry.name.c_str(),
			entry.label.c_str(),
			y_pred);
	}

	return 0;
}
