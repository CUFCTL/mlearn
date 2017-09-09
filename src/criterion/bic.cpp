/**
 * @file criterion/bic.cpp
 *
 * Implementation of the BIC layer.
 */
#include "criterion/bic.h"
#include "util/logger.h"

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

/**
 * Print a BIC layer.
 */
void BICLayer::print() const
{
	log(LL_VERBOSE, "BIC");
}

}
