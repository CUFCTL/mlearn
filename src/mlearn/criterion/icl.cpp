/**
 * @file clustering/icl.cpp
 *
 * Implementation of the ICL layer.
 */
#include "mlearn/criterion/icl.h"
#include "mlearn/util/logger.h"



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
	float L = layer->log_likelihood();
	float E = layer->entropy();

	return log(n) * k - 2 * L - 2 * E;
}



/**
 * Print an ICL layer.
 */
void ICLLayer::print() const
{
	Logger::log(LogLevel::Verbose, "ICL");
}



}
