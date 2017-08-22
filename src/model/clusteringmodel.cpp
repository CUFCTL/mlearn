/**
 * @file model/clusteringmodel.cpp
 *
 * Implementation of the clustering model.
 */
#include <iomanip>
#include "model/clusteringmodel.h"
#include "util/logger.h"
#include "util/timer.h"

namespace ML {

/**
 * Construct a clustering model.
 *
 * @param layers
 */
ClusteringModel::ClusteringModel(const std::vector<ClusteringLayer *>& layers)
{
	// initialize layers
	this->_layers = layers;
}

/**
 * Perform clustering on a dataset.
 *
 * @param input
 */
std::vector<int> ClusteringModel::run(const Dataset& input)
{
	timer_push("Clustering");

	log(LL_VERBOSE, "Input: %d samples, %d classes",
		input.entries().size(),
		input.labels().size());
	log(LL_VERBOSE, "");

	// get data matrix X
	Matrix X = input.load_data();

	// perform clustering on X
	for ( ClusteringLayer *layer : this->_layers ) {
		layer->compute(X);
		layer->print();
	}

	// select the best model
	int index = 0;

	log(LL_INFO, "selecting model %d", index);

	return this->_layers[index]->output();
}

}
