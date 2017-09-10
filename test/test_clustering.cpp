/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <iostream>
#include <memory>
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

	// load input data
	Dataset input_data(nullptr, args.path_input);

	// construct feature layer
	std::unique_ptr<FeatureLayer> feature;

	if ( args.feature == "identity" ) {
		feature.reset(new IdentityLayer());
	}
	else if ( args.feature == "pca" ) {
		feature.reset(new PCALayer());
	}
	else if ( args.feature == "lda" ) {
		feature.reset(new LDALayer());
	}
	else if ( args.feature == "ica" ) {
		feature.reset(new ICALayer());
	}
	else {
		std::cerr << "error: feature must be identity | pca | lda | ica\n";
		exit(1);
	}

	// construct clustering layer
	std::unique_ptr<ClusteringLayer> clustering;

	if ( args.clustering == "gmm" ) {
		clustering.reset(new GMMLayer(args.k));
	}
	else if ( args.clustering == "k-means" ) {
		clustering.reset(new KMeansLayer(args.k));
	}
	else {
		std::cerr << "error: clustering must be 'gmm' or 'k-means'\n";
		exit(1);
	}

	// construct criterion layer
	std::unique_ptr<CriterionLayer> criterion;

	if ( args.criterion == "bic" ) {
		criterion.reset(new BICLayer());
	}
	else if ( args.criterion == "icl" ) {
		criterion.reset(new ICLLayer());
	}
	else {
		std::cerr << "error: criterion must be 'bic' or 'icl'\n";
		exit(1);
	}

	// create clustering model
	ClusteringModel model(feature.get(), clustering.get());

	// extract features from input data
	model.extract(input_data);

	// perform clustering on input data
	std::vector<int> Y_pred = model.predict();

	// print clustering results
	model.validate(Y_pred);
	model.print_results(Y_pred);

	// evaluate criterion of clustering model
	float value = criterion->compute(clustering.get());

	log(LL_VERBOSE, "criterion value: %f", value);
	log(LL_VERBOSE, "");

	// print timing results
	timer_print();

	gpu_finalize();

	return 0;
}
