/**
 * @file criterion/bic.cpp
 *
 * Implementation of the BIC layer.
 */
#include "mlearn/criterion/bic.h"
#include "mlearn/util/logger.h"



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
	float L = layer->log_likelihood();

	return logf(n) * k - 2 * L;
}



/**
 * Print a BIC layer.
 */
void BICLayer::print() const
{
	Logger::log(LogLevel::Verbose, "BIC");
}



}
