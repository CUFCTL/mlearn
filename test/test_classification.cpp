/**
 * @file test_classification.cpp
 *
 * Test suite for the classification model.
 */
#include <cstdlib>
#include <iostream>
#include <mlearn.h>

using namespace ML;

int main(int argc, char **argv)
{
	gpu_init();

	if ( argc != 1 ) {
		std::cerr << "usage: ./test-classification\n";
		exit(1);
	}

	const char *train_filename = "test/data/iris.train";
	const char *test_filename = "test/data/iris.test";

	// set loglevel
	LOGLEVEL = LL_VERBOSE;

	// load train set, test set
	Dataset train_set(nullptr, train_filename);
	Dataset test_set(nullptr, test_filename, false);

	// create clustering model
	FeatureLayer *feature = new PCALayer(-1);
	ClassifierLayer *classifier = new KNNLayer(1, m_dist_L1);
	ClassificationModel model(feature, classifier);

	// train the model with the training set
	train_set.print();
	model.train(train_set);

	// classify the test set with the trained model
	test_set.print();
	std::vector<DataLabel> Y_pred = model.predict(test_set);

	// print results
	log(LL_INFO, "Results");

	for ( size_t i = 0; i < test_set.entries().size(); i++ ) {
		const DataLabel& y_pred = Y_pred[i];
		const DataEntry& entry = test_set.entries()[i];

		log(LL_INFO, "%-12s -> %-4s", entry.name.c_str(), y_pred.c_str());
	}
	log(LL_INFO, "");

	// print timing results
	timer_print();

	gpu_finalize();

	return 0;
}
