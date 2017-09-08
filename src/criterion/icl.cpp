/**
 * @file clustering/icl.cpp
 *
 * Implementation of the ICL layer.
 */
#include "criterion/icl.h"

namespace ML {

/**
 * Compute the Integrated Completed Likelihood of a model:
 *
 *   ICL = ln(n) * k - 2 * ln(L) - 2 * E
 *
 * @param layer
 */
float ICLLayer::compute(ClusteringLayer *layer)
{
	int k = layer->num_parameters();
	int n = layer->num_samples();
	precision_t L = layer->log_likelihood();
	precision_t E = layer->entropy();

	return logf(n) * k - 2 * L - 2 * E;
}

}
