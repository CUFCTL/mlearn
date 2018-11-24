/**
 * @file test_classification.cpp
 *
 * Test suite for the classification model.
 */
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>
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
		"  --type TYPE        data type ([csv], image, genome)\n"
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

	if ( args.data_type == "image" )
	{
		data_iter.reset(new ImageIterator(args.data_path));
	}
	else if ( args.data_type == "genome" )
	{
		data_iter.reset(new GenomeIterator(args.data_path));
	}
	else if ( args.data_type == "csv" )
	{
		data_iter.reset(new CSVIterator(args.data_path));
	}
	else
	{
		std::cerr << "error: type must be csv | image | genome\n";
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

	// construct feature layer
	std::unique_ptr<TransformerLayer> feature;

	if ( args.feature == "identity" )
	{
		feature.reset();
	}
	else if ( args.feature == "pca" )
	{
		feature.reset(new PCALayer());
	}
	else if ( args.feature == "lda" )
	{
		feature.reset(new LDALayer());
	}
	else if ( args.feature == "ica" )
	{
		feature.reset(new ICALayer());
	}
	else
	{
		std::cerr << "error: feature must be identity | pca | lda | ica\n";
		exit(1);
	}

	// construct classifier layer
	std::unique_ptr<EstimatorLayer> classifier;

	if ( args.classifier == "knn" )
	{
		classifier.reset(new KNNLayer());
	}
	else if ( args.classifier == "bayes" )
	{
		classifier.reset(new BayesLayer());
	}
	else
	{
		std::cerr << "error: classifier must be 'knn' or 'bayes'\n";
		exit(1);
	}

	// create classification model
	ClassificationModel model(feature.get(), classifier.get());

	model.print();

	// fit model to training set
	model.fit(X_train, y_train, dataset.classes().size());

	// perform classification on test set
	std::vector<int> y_pred = model.predict(X_test);

	// compute test accuracy
	float error_rate = model.score(y_test, y_pred);

	Logger::log(LogLevel::Verbose, "Test error: %.3f", error_rate);
	Logger::log(LogLevel::Verbose, "");

	// print classification results on entire dataset
	Logger::log(LogLevel::Verbose, "Results");

	std::vector<int> y_pred_all = model.predict(X);

	for ( size_t i = 0; i < y_pred_all.size(); i++ )
	{
		const std::string& name = dataset.entries()[i].name;
		const std::string& label_true = dataset.classes()[y[i]];
		const std::string& label_pred = dataset.classes()[y_pred_all[i]];

		const char *s = (y_pred_all[i] != y[i])
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
