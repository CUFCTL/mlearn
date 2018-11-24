/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering model.
 */
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mlearn.h>



using namespace mlearn;



typedef struct {
	std::string data_path;
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
		"  --loglevel LEVEL   log level (0=error, 1=warn, [2]=info, 3=verbose, 4=debug)\n"
		"  --dataset PATH     path to dataset ([data/iris.txt])\n"
		"  --type TYPE        data type ([csv], image, genome)\n"
		"  --clus CLUSTERING  clustering method ([kmeans], gmm)\n"
		"  --min-k K          minimum number of clusters [1]\n"
		"  --max-k K          maximum number of clusters [5]\n"
		"  --crit CRITERION   model selection criterion ([bic], icl)\n";
}



args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"data/iris.txt",
		"csv",
		"kmeans", 1, 5,
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
			Device::initialize();
			break;
		case 'e':
			Logger::LEVEL = (LogLevel) atoi(optarg);
			break;
		case 'p':
			args.data_path = optarg;
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
	Random::seed();

	// construct data iterator
	DataIterator *data_iter;

	if ( args.data_type == "image" ) {
		data_iter = new ImageIterator(args.data_path);
	}
	else if ( args.data_type == "genome" ) {
		data_iter = new GenomeIterator(args.data_path);
	}
	else if ( args.data_type == "csv" ) {
		data_iter = new CSVIterator(args.data_path);
	}
	else {
		std::cerr << "error: type must be image | genome | csv\n";
		exit(1);
	}

	// load dataset
	Dataset dataset(data_iter);

	// construct clustering layers
	std::vector<ClusteringLayer *> clustering;

	for ( int k = args.min_k; k <= args.max_k; k++ ) {
		if ( args.clustering == "gmm" ) {
			clustering.push_back(new GMMLayer(k));
		}
		else if ( args.clustering == "kmeans" ) {
			clustering.push_back(new KMeansLayer(k));
		}
		else {
			std::cerr << "error: clustering must be 'gmm' or 'kmeans'\n";
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

	model.print();

	// perform clustering on input data
	Matrix X = dataset.load_data();
	std::vector<Matrix> X_col = m_copy_columns(X);
	model.fit(X_col);

	std::vector<int> y_pred = model.predict(X_col);
	float error_rate = model.score(dataset, y_pred);

	// print clustering results
	Logger::log(LogLevel::Verbose, "Results");

	for ( size_t i = 0; i < dataset.entries().size(); i++ ) {
		const DataEntry& entry = dataset.entries()[i];

		Logger::log(LogLevel::Verbose, "%-4s (%s) -> %d",
			entry.name.c_str(),
			entry.label.c_str(),
			y_pred[i]);
	}

	Logger::log(LogLevel::Verbose, "Error rate: %.3f", error_rate);
	Logger::log(LogLevel::Verbose, "");

	// print timing results
	Timer::print();

	return 0;
}
