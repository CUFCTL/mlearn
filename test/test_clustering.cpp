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
	std::string data_type;
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
		"  --gpu              enable GPU acceleration\n"
		"  --loglevel LEVEL   log level ([1]=info, 2=verbose, 3=debug)\n"
		"  --path PATH        path to dataset (default is IRIS dataset)\n"
		"  --type TYPE        data type ([none], image, genome)\n"
		"  --feat FEATURE     feature extraction method ([identity], pca, lda, ica)\n"
		"  --clus CLUSTERING  clustering method ([k-means], gmm)\n"
		"  --crit CRITERION   model selection criterion ([bic], icl)\n"
		"  --k K              number of clusters\n";
}

args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"test/data/iris.train",
		"none",
		"identity",
		"k-means",
		"bic",
		0
	};

	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ "path", required_argument, 0, 'p' },
		{ "type", required_argument, 0, 'd' },
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
		case 'p':
			args.path_input = optarg;
			break;
		case 'd':
			args.data_type = optarg;
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

	return args;
}

int main(int argc, char **argv)
{
	// parse command-line arguments
	args_t args = parse_args(argc, argv);

	// initialize random number engine
	RNG_seed();

	// initialize GPU if enabled
	gpu_init();

	// construct data iterator
	std::unique_ptr<DataIterator> data_iter;

	if ( args.data_type == "image" ) {
		data_iter.reset(new Image());
	}
	else if ( args.data_type == "genome" ) {
		data_iter.reset(new Genome());
	}
	else if ( args.data_type == "none" ) {
		data_iter.reset(nullptr);
	}
	else {
		std::cerr << "error: type must be image | genome | none\n";
		exit(1);
	}

	// load input data
	Dataset input_data(data_iter.get(), args.path_input);

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
