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
	std::string method;
	std::string criterion;
	int k;
} args_t;

int main(int argc, char **argv)
{
	// parse command-line arguments
	if ( argc != 4 ) {
		std::cerr << "usage: ./test-clustering [method] [criterion] [k]\n";
		exit(1);
	}

	args_t args = {
		"test/data/iris.train",
		argv[1],
		argv[2],
		atoi(argv[3])
	};

	GPU = true;
	LOGLEVEL = LL_VERBOSE;

	// initialize GPU if enabled
	gpu_init();

	// load input dataset
	Dataset input_data(nullptr, args.filename);

	// construct clustering layer
	std::vector<ClusteringLayer *> layers;

	if ( args.method == "gmm" ) {
		layers.push_back(new GMMLayer(args.k));
	}
	else if ( args.method == "k-means" ) {
		layers.push_back(new KMeansLayer(args.k));
	}
	else {
		std::cerr << "error: method must be 'gmm' or 'k-means'\n";
		exit(1);
	}

	// construct criterion layer
	CriterionLayer *criterion;

	if ( args.criterion == "bic" ) {
		criterion = new BICLayer();
	}
	else if ( args.criterion == "icl" ) {
		criterion = new ICLLayer();
	}
	else {
		std::cerr << "error: criterion must be 'bic' or 'icl'\n";
		exit(1);
	}

	// perform clustering on input data
	ClusteringModel model(layers, criterion);

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
