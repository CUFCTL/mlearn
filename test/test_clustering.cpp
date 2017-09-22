/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mlearn.h>

using namespace ML;

typedef struct {
	std::string path_input;
	std::string data_type;
	std::string clustering;
	int min_k;
	int max_k;
	std::string criterion;
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
		"  --clus CLUSTERING  clustering method ([k-means], gmm)\n"
		"  --min-k K          minimum number of clusters [1]\n"
		"  --max-k K          maximum number of clusters [5]\n"
		"  --crit CRITERION   model selection criterion ([bic], icl)\n";
}

args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"test/data/iris.train",
		"none",
		"k-means", 1, 5,
		"bic",
	};

	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ "path", required_argument, 0, 'p' },
		{ "type", required_argument, 0, 'd' },
		{ "clus", required_argument, 0, 'c' },
		{ "min-k", required_argument, 0, 'i' },
		{ "max-k", required_argument, 0, 'a' },
		{ "crit", required_argument, 0, 'r' },
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
		case 'c':
			args.clustering = optarg;
			break;
		case 'i':
			args.min_k = atoi(optarg);
			break;
		case 'a':
			args.max_k = atoi(optarg);
			break;
		case 'r':
			args.criterion = optarg;
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	if ( args.min_k > args.max_k ) {
		std::cerr << "error: min-k must be less than or equal to max-k\n";
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
	DataIterator *data_iter;

	if ( args.data_type == "image" ) {
		data_iter = new Image();
	}
	else if ( args.data_type == "genome" ) {
		data_iter = new Genome();
	}
	else if ( args.data_type == "none" ) {
		data_iter = nullptr;
	}
	else {
		std::cerr << "error: type must be image | genome | none\n";
		exit(1);
	}

	// load input data
	Dataset input_data(data_iter, args.path_input);

	// construct clustering layers
	std::vector<ClusteringLayer *> clustering;

	for ( int k = args.min_k; k <= args.max_k; k++ ) {
		if ( args.clustering == "gmm" ) {
			clustering.push_back(new GMMLayer(k));
		}
		else if ( args.clustering == "k-means" ) {
			clustering.push_back(new KMeansLayer(k));
		}
		else {
			std::cerr << "error: clustering must be 'gmm' or 'k-means'\n";
			exit(1);
		}
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

	// create clustering model
	ClusteringModel model(clustering, criterion);

	// perform clustering on input data
	Matrix X = input_data.load_data();
	model.predict(X);

	std::vector<int> Y_pred = model.best_layer()->output();

	// print clustering results
	log(LL_VERBOSE, "Best clustering model:");
	log(LL_VERBOSE, "");

	model.best_layer()->print();
	log(LL_VERBOSE, "");

	model.validate(input_data, Y_pred);
	model.print_results(input_data, Y_pred);

	// print timing results
	timer_print();

	gpu_finalize();

	return 0;
}
