/**
 * @file clustering/criterion.cpp
 *
 * Implementation of the abstract criterion layer.
 */
#include "mlearn/criterion/criterion.h"
#include "mlearn/util/logger.h"



namespace ML {



/**
 * Select the best model from a list of models; that is,
 * the model with the lowest criterion value.
 *
 * @param layer
 */
ClusteringLayer * CriterionLayer::select(const std::vector<ClusteringLayer *>& layers)
{
	ClusteringLayer *min_layer = nullptr;
	float min_value = INFINITY;

	for ( size_t i = 0; i < layers.size(); i++ ) {
		if ( layers[i]->success() ) {
			float value = score(layers[i]);

			if ( value < min_value ) {
				min_layer = layers[i];
				min_value = value;
			}

			Logger::log(LogLevel::Verbose, "model %d: %8.3f", i + 1, value);
		}
		else {
			Logger::log(LogLevel::Verbose, "model %d: FAILED", i + 1);
		}
	}
	Logger::log(LogLevel::Verbose, "");

	if ( min_layer == nullptr ) {
		Logger::log(LogLevel::Warn, "warning: all models failed");
	}

	return min_layer;
}



}
