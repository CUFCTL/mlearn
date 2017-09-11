/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <getopt.h>
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

void print_usage()
{
	std::cerr <<
		"Usage: ./test-clustering [options]\n"
		"\n"
		"Options:\n"
		"  --gpu              use GPU acceleration\n"
		"  --loglevel LEVEL   set the log level (1=info, 2=verbose, 3=debug)\n"
		"  --feat FEATURE     specify feature extraction ([identity], pca, lda, ica)\n"
		"  --clus CLUSTERING  specify clustering method ([k-means], gmm)\n"
		"  --crit CRITERION   specify model selection criterion ([bic], icl)\n"
		"  --k K              specify number of clusters\n";
}

int main(int argc, char **argv)
{
	// parse command-line arguments
	args_t args = {
		"test/data/iris.train",
		"identity",
		"k-means",
		"bic",
		0
	};

	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ "feat", required_argument, 0, 'f' },
		{ "clus", required_argument, 0, 'c' },
		{ "crit", required_argument, 0, 'r' },
		{ "k", required_argument, 0, 'k' },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, nullptr)) != -1 ) {
		switch ( opt ) {
		case 'g':
			GPU = true;
			break;
		case 'e':
			LOGLEVEL = (logger_level_t) atoi(optarg);
			break;
		case 'f':
			args.feature = optarg;
			break;
		case 'c':
			args.clustering = optarg;
			break;
		case 'r':
			args.criterion = optarg;
			break;
		case 'k':
			args.k = atoi(optarg);
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	if ( args.k == 0 ) {
		std::cerr << "error: --k is required\n";
		print_usage();
		exit(1);
	}

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
