/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <iostream>
#include <mlearn.h>

using namespace ML;

typedef struct {
	std::string filename;
	std::string layer_type;
	int k;
} args_t;

int main(int argc, char **argv)
{
	gpu_init();

	// parse command-line arguments
	if ( argc < 3 ) {
		std::cerr << "usage: ./test-clustering [method] [k]\n";
		exit(1);
	}

	args_t args = {
		"test/data/iris.train",
		argv[1],
		atoi(argv[2])
	};

	// set loglevel
	LOGLEVEL = LL_VERBOSE;

	// load input dataset
	Dataset input_data(nullptr, args.filename);

	// create clustering model
	std::vector<ClusteringLayer *> layers;

	if ( args.layer_type == "gmm" ) {
		layers.push_back(new GMMLayer(args.k));
	}
	else if ( args.layer_type == "k-means" ) {
		layers.push_back(new KMeansLayer(args.k));
	}
	else {
		std::cerr << "error: method must be 'gmm' or 'k-means'\n";
		exit(1);
	}

	CriterionLayer *criterion = new BICLayer();

	ClusteringModel model(layers, criterion);

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
	log(LL_INFO, "");

	// print timing results
	timer_print();

	gpu_finalize();

	return 0;
}
