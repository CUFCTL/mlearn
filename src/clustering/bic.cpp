/**
 * @file clustering/bic.cpp
 *
 * Implementation of the BIC layer.
 */
#include "clustering/bic.h"

namespace ML {

/**
 * Compute the Bayesian information criterion of a model:
 *
 *   BIC = ln(n) * k - 2 * ln(L)
 *
 * @param layer
 */
float BICLayer::compute(ClusteringLayer *layer)
{
	int k = layer->num_parameters();
	int n = layer->num_samples();
	precision_t L = layer->log_likelihood();

	return logf(n) * k - 2 * L;
}

}
