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



using namespace ML;



typedef struct {
	std::string path_train;
	std::string path_test;
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
		"  --path_train PATH  path to training set [iris.train]\n"
		"  --path_test PATH   path to test set [iris.test]\n"
		"  --type TYPE        data type ([csv], image, genome)\n"
		"  --feat FEATURE     feature extraction method ([identity], pca, lda, ica)\n"
		"  --clas CLASSIFIER  classification method ([knn], bayes)\n";
}



args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"data/iris.train",
		"data/iris.test",
		"csv",
		"identity",
		"knn"
	};

	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ "path_train", required_argument, 0, 't' },
		{ "path_test", required_argument, 0, 'r' },
		{ "type", required_argument, 0, 'd' },
		{ "feat", required_argument, 0, 'f' },
		{ "clas", required_argument, 0, 'c' },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, nullptr)) != -1 ) {
		switch ( opt ) {
		case 'g':
			GPU = true;
			break;
		case 'e':
			Logger::LEVEL = (LogLevel) atoi(optarg);
			break;
		case 't':
			args.path_train = optarg;
			break;
		case 'r':
			args.path_test = optarg;
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

	// initialize GPU if enabled
	gpu_init();

	// construct data iterators
	std::unique_ptr<DataIterator> train_iter;
	std::unique_ptr<DataIterator> test_iter;

	if ( args.data_type == "image" ) {
		train_iter.reset(new ImageIterator(args.path_train));
		test_iter.reset(new ImageIterator(args.path_test));
	}
	else if ( args.data_type == "genome" ) {
		train_iter.reset(new GenomeIterator(args.path_train));
		test_iter.reset(new GenomeIterator(args.path_test));
	}
	else if ( args.data_type == "csv" ) {
		train_iter.reset(new CSVIterator(args.path_train));
		test_iter.reset(new CSVIterator(args.path_test));
	}
	else {
		std::cerr << "error: type must be image | genome | csv\n";
		exit(1);
	}

	// load train set, test set
	Dataset train_set(train_iter.get());
	Dataset test_set(test_iter.get());

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

	// construct classifier layer
	std::unique_ptr<ClassifierLayer> classifier;

	if ( args.classifier == "knn" ) {
		classifier.reset(new KNNLayer());
	}
	else if ( args.classifier == "bayes" ) {
		classifier.reset(new BayesLayer());
	}
	else {
		std::cerr << "error: classifier must be 'knn' or 'bayes'\n";
		exit(1);
	}

	// create classification model
	ClassificationModel model(feature.get(), classifier.get());

	model.print();

	// fit model to training set
	model.fit(train_set);

	// perform classification on test set
	std::vector<int> y_pred = model.predict(test_set);

	// print classification results
	model.validate(test_set, y_pred);
	model.print_results(test_set, y_pred);

	// print timing results
	Timer::print();

	gpu_finalize();

	return 0;
}
