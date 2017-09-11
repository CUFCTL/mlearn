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
		"  --loglevel LEVEL   log level ([1]=info, 2=verbose, 3=debug)\n"
		"  --path_train PATH  path to training set [iris.train]\n"
		"  --path_test PATH   path to test set [iris.test]\n"
		"  --type TYPE        data type ([none], image, genome)\n"
		"  --feat FEATURE     feature extraction method ([identity], pca, lda, ica)\n"
		"  --clas CLASSIFIER  classification method ([knn], bayes)\n";
}

args_t parse_args(int argc, char **argv)
{
	args_t args = {
		"test/data/iris.train",
		"test/data/iris.test",
		"none",
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
		{ "clus", required_argument, 0, 'c' },
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

	// load train set, test set
	Dataset train_set(data_iter.get(), args.path_train);
	Dataset test_set(data_iter.get(), args.path_test);

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

	// extract features from training set
	model.train(train_set);

	// perform classification on test set
	std::vector<DataLabel> Y_pred = model.predict(test_set);

	// print classification results
	model.validate(test_set, Y_pred);
	model.print_results(test_set, Y_pred);

	// print timing results
	timer_print();

	gpu_finalize();

	return 0;
}
