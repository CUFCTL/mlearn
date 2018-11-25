/**
 * @file test_classification.cpp
 *
 * Test suite for the classifier pipeline.
 */
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mlearn.h>



using namespace mlearn;



typedef struct
{
	std::string data_path;
	std::string data_type;
	std::string feature;
	std::string classifier;
} args_t;



void print_usage()
{
	std::cerr <<
		"Usage: ./test-classification [options]\n"
		"\n"
		"Options:\n"
		"  --gpu              enable GPU acceleration\n"
		"  --loglevel LEVEL   log level (0=error, 1=warn, [2]=info, 3=verbose, 4=debug)\n"
		"  --dataset PATH     path to dataset [data/iris.txt]\n"
		"  --type TYPE        data type ([csv], genome, image)\n"
		"  --feat FEATURE     feature extraction method ([identity], pca, lda, ica)\n"
		"  --clas CLASSIFIER  classification method ([knn], bayes)\n";
}



args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"data/iris.txt",
		"csv",
		"identity",
		"knn"
	};

	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ "dataset", required_argument, 0, 't' },
		{ "type", required_argument, 0, 'd' },
		{ "feat", required_argument, 0, 'f' },
		{ "clas", required_argument, 0, 'c' },
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
		case 't':
			args.data_path = optarg;
			break;
		case 'd':
			args.data_type = optarg;
			break;
		case 'f':
			args.feature = optarg;
			break;
		case 'c':
			args.classifier = optarg;
			break;
		case '?':
			print_usage();
			exit(1);
		}
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
	std::unique_ptr<DataIterator> data_iter;

	if ( args.data_type == "csv" )
	{
		data_iter.reset(new CSVIterator(args.data_path));
	}
	else if ( args.data_type == "genome" )
	{
		data_iter.reset(new GenomeIterator(args.data_path));
	}
	else if ( args.data_type == "image" )
	{
		data_iter.reset(new ImageIterator(args.data_path));
	}
	else
	{
		std::cerr << "error: type must be csv | genome | image\n";
		exit(1);
	}

	// load dataset
	Dataset dataset(data_iter.get());

	Matrix X = dataset.load_data();
	std::vector<int> y = dataset.labels();

	Logger::log(LogLevel::Verbose, "Dataset: %d samples, %d classes",
		dataset.entries().size(),
		dataset.classes().size());
	Logger::log(LogLevel::Verbose, "");

	// create train set and test set
	Matrix X_train;
	Matrix X_test;
	std::vector<int> y_train;
	std::vector<int> y_test;

	Dataset::train_test_split(X, y, 0.2, X_train, y_train, X_test, y_test);

	// construct transformer layers
	std::vector<TransformerLayer*> transforms;

	transforms.push_back(new Scaler(true, false));

	if ( args.feature == "identity" )
	{
		// do nothing
	}
	else if ( args.feature == "pca" )
	{
		transforms.push_back(new PCALayer());
	}
	else if ( args.feature == "lda" )
	{
		transforms.push_back(new LDALayer());
	}
	else if ( args.feature == "ica" )
	{
		transforms.push_back(new ICALayer());
	}
	else
	{
		std::cerr << "error: feature must be identity | pca | lda | ica\n";
		exit(1);
	}

	// construct classifier layer
	EstimatorLayer* classifier;

	if ( args.classifier == "knn" )
	{
		classifier = new KNNLayer();
	}
	else if ( args.classifier == "bayes" )
	{
		classifier = new BayesLayer();
	}
	else
	{
		std::cerr << "error: classifier must be 'knn' or 'bayes'\n";
		exit(1);
	}

	// create classifier pipeline
	Pipeline pipeline(transforms, classifier);

	pipeline.print();

	// fit pipeline to training set
	pipeline.fit(X_train, y_train, dataset.classes().size());

	// evaluate pipeline on test set
	float accuracy = pipeline.score(X_test, y_test);

	Logger::log(LogLevel::Verbose, "Test accuracy: %.3f", accuracy);
	Logger::log(LogLevel::Verbose, "");

	// print classification results on entire dataset
	Logger::log(LogLevel::Verbose, "Results");

	std::vector<int> y_pred = pipeline.predict(X);

	for ( size_t i = 0; i < y_pred.size(); i++ )
	{
		const std::string& name = dataset.entries()[i].name;
		const std::string& label_true = dataset.classes()[y[i]];
		const std::string& label_pred = dataset.classes()[y_pred[i]];

		const char *s = (y_pred[i] != y[i])
			? "(!)"
			: "";

		Logger::log(LogLevel::Verbose, "%-12s (%-4s) -> %-4s %s",
			name.c_str(),
			label_true.c_str(),
			label_pred.c_str(),
			s);
	}

	// print timing results
	Timer::print();

	return 0;
}
