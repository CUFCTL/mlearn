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
 * @param criterion
 */
ClusteringModel::ClusteringModel(const std::vector<ClusteringLayer *>& layers, CriterionLayer *criterion)
{
	// initialize layers
	this->_layers = layers;
	this->_criterion = criterion;
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

	// run and evaluate each clustering layer
	std::vector<float> values;

	for ( ClusteringLayer *layer : this->_layers ) {
		layer->compute(X);
		layer->print();

		float value = this->_criterion->compute(layer);
		values.push_back(value);

		log(LL_INFO, "criterion value: %f", value);
		log(LL_INFO, "");
	}

	// select the layer with the lowest criterion value
	size_t min_index = 0;

	for ( size_t i = 0; i < values.size(); i++ ) {
		if ( values[i] < values[min_index] ) {
			min_index = i;
		}
	}

	log(LL_INFO, "selecting model %d", min_index);

	timer_pop();

	return this->_layers[min_index]->output();
}

}
