/**
 * @file test_classification.cpp
 *
 * Test suite for the classification model.
 */
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mlearn.h>

using namespace ML;

typedef struct {
	std::string path_train;
	std::string path_test;
	std::string feature;
	std::string classifier;
} args_t;

int main(int argc, char **argv)
{
	// parse command-line arguments
	if ( argc != 3 ) {
		std::cerr << "usage: ./test-classification [feature] [classifier]\n";
		exit(1);
	}

	args_t args = {
		"test/data/iris.train",
		"test/data/iris.test",
		argv[1],
		argv[2]
	};

	GPU = true;
	LOGLEVEL = LL_VERBOSE;

	// initialize GPU if enabled
	gpu_init();

	// load train set, test set
	Dataset train_set(nullptr, args.path_train);
	Dataset test_set(nullptr, args.path_test, false);

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
