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
 * @param feature
 * @param layers
 * @param criterion
 */
ClusteringModel::ClusteringModel(FeatureLayer *feature, const std::vector<ClusteringLayer *>& layers, CriterionLayer *criterion)
{
	// initialize layers
	this->_feature = feature;
	this->_layers = layers;
	this->_criterion = criterion;

	// log hyperparameters
	log(LL_VERBOSE, "Hyperparameters");

	this->_feature->print();

	for ( ClusteringLayer *layer : this->_layers ) {
		layer->print();
	}

	this->_criterion->print();

	log(LL_VERBOSE, "");
}

/**
 * Extract features from input data.
 *
 * @param input
 */
void ClusteringModel::extract(const Dataset& input)
{
	timer_push("Feature extraction");

	this->_input = input;

	log(LL_VERBOSE, "Input data: %d samples, %d classes",
		input.entries().size(),
		input.labels().size());

	// get data matrix X
	Matrix X = input.load_data();

	// subtract mean from X
	Matrix mu = X.mean_column();

	X.subtract_columns(mu);

	// project X into feature space
	this->_feature->compute(X, this->_input.entries(), this->_input.labels().size());
	this->_P = this->_feature->project(X);

	timer_pop();

	log(LL_VERBOSE, "");
}

/**
 * Perform clustering on the input data.
 */
std::vector<int> ClusteringModel::predict()
{
	timer_push("Clustering");

	// run and evaluate each clustering layer
	std::vector<float> values;

	for ( ClusteringLayer *layer : this->_layers ) {
		layer->compute(this->_P);

		float value = this->_criterion->compute(layer);
		values.push_back(value);

		log(LL_VERBOSE, "criterion value: %f", value);
		log(LL_VERBOSE, "");
	}

	// select the layer with the lowest criterion value
	size_t min_index = 0;

	for ( size_t i = 0; i < values.size(); i++ ) {
		if ( values[i] < values[min_index] ) {
			min_index = i;
		}
	}

	log(LL_VERBOSE, "selecting model %d", min_index);
	log(LL_VERBOSE, "");

	timer_pop();

	return this->_layers[min_index]->output();
}

}
