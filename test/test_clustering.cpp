/**
 * @file test_clustering.cpp
 *
 * Test suite for the clustering pipeline.
 */
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <map>
#include <mlearn.h>



using namespace mlearn;



typedef struct
{
	std::string data_path;
	std::string data_type;
	std::string clustering;
	int min_k;
	int max_k;
	Criterion criterion;
} args_t;



const std::map<std::string, Criterion> CRITERION_NAMES = {
        { "aic", Criterion::AIC },
        { "bic", Criterion::BIC },
        { "icl", Criterion::ICL }
};



void print_usage()
{
	std::cerr <<
		"Usage: ./test-clustering [options]\n"
		"\n"
		"Options:\n"
		"  --gpu              enable GPU acceleration\n"
		"  --loglevel LEVEL   log level (0=error, 1=warn, [2]=info, 3=verbose, 4=debug)\n"
		"  --dataset PATH     path to dataset ([data/iris.txt])\n"
		"  --type TYPE        data type ([csv], genome, image)\n"
		"  --clus CLUSTERING  clustering method ([kmeans], gmm)\n"
		"  --min-k K          minimum number of clusters [1]\n"
		"  --max-k K          maximum number of clusters [5]\n"
		"  --crit CRITERION   model selection criterion (aic, [bic], icl)\n";
}



args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"data/iris.txt",
		"csv",
		"kmeans", 1, 5,
		Criterion::BIC,
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
	while ( (opt = getopt_long_only(argc, argv, "", long_options, nullptr)) != -1 )
	{
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
			try
			{
				args.criterion = CRITERION_NAMES.at(optarg);
			}
			catch ( std::exception& e )
			{
				std::cerr << "error: criterion must be aic | bic | icl\n";
				print_usage();
				exit(1);
			}
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	if ( args.min_k > args.max_k )
	{
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

	if ( args.data_type == "csv" )
	{
		data_iter = new CSVIterator(args.data_path);
	}
	else if ( args.data_type == "genome" )
	{
		data_iter = new GenomeIterator(args.data_path);
	}
	else if ( args.data_type == "image" )
	{
		data_iter = new ImageIterator(args.data_path);
	}
	else
	{
		std::cerr << "error: type must be csv | genome | image\n";
		exit(1);
	}

	// load dataset
	Dataset dataset(data_iter);
	Matrix X = dataset.load_data();
	std::vector<int> y = dataset.labels();

	// construct clustering models
	std::vector<ClusteringLayer *> models;

	for ( int k = args.min_k; k <= args.max_k; k++ )
	{
		if ( args.clustering == "gmm" )
		{
			models.push_back(new GMMLayer(k));
		}
		else if ( args.clustering == "kmeans" )
		{
			models.push_back(new KMeansLayer(k));
		}
		else
		{
			std::cerr << "error: clustering must be 'gmm' or 'kmeans'\n";
			exit(1);
		}
	}

	// construct criterion layer
	CriterionLayer criterion(args.criterion, models);

	// create clustering pipeline
	Pipeline pipeline({}, &criterion);

	pipeline.print();

	// perform clustering on dataset
	pipeline.fit(X);

	std::vector<int> y_pred = pipeline.predict(X);
	float purity = pipeline.score(X, y);

	// print clustering results
	Logger::log(LogLevel::Verbose, "Results");

	for ( size_t i = 0; i < dataset.entries().size(); i++ )
	{
		const DataEntry& entry = dataset.entries()[i];

		Logger::log(LogLevel::Verbose, "%-4s (%s) -> %d",
			entry.name.c_str(),
			entry.label.c_str(),
			y_pred[i]);
	}

	Logger::log(LogLevel::Verbose, "Purity: %.3f", purity);
	Logger::log(LogLevel::Verbose, "");

	// print timing results
	Timer::print();

	return 0;
}
