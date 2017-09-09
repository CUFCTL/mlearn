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
	std::string path_input;
	std::string feature;
	std::string clustering;
	std::string criterion;
	int k;
} args_t;

int main(int argc, char **argv)
{
	// parse command-line arguments
	if ( argc != 5 ) {
		std::cerr << "usage: ./test-clustering [feature] [clustering] [criterion] [k]\n";
		exit(1);
	}

	args_t args = {
		"test/data/iris.train",
		argv[1],
		argv[2],
		argv[3],
		atoi(argv[4])
	};

	GPU = true;
	LOGLEVEL = LL_VERBOSE;

	// initialize GPU if enabled
	gpu_init();

	// load input dataset
	Dataset input_data(nullptr, args.path_input);

	// construct feature layer
	FeatureLayer *feature;

	if ( args.feature == "identity" ) {
		feature = new IdentityLayer();
	}
	else if ( args.feature == "pca" ) {
		feature = new PCALayer();
	}
	else if ( args.feature == "lda" ) {
		feature = new LDALayer();
	}
	else if ( args.feature == "ica" ) {
		feature = new ICALayer();
	}
	else {
		std::cerr << "error: feature must be identity | pca | lda | ica\n";
		exit(1);
	}

	// construct clustering layer
	std::vector<ClusteringLayer *> layers;

	if ( args.clustering == "gmm" ) {
		layers.push_back(new GMMLayer(args.k));
	}
	else if ( args.clustering == "k-means" ) {
		layers.push_back(new KMeansLayer(args.k));
	}
	else {
		std::cerr << "error: clustering must be 'gmm' or 'k-means'\n";
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
	ClusteringModel model(feature, layers, criterion);

	model.extract(input_data);
	std::vector<int> Y_pred = model.predict();

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
