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
 * @param clusters
 * @param clustering
 */
ClusteringModel::ClusteringModel(const std::vector<int>& clusters, ClusteringLayer *clustering)
{
	// initialize hyperparameters
	this->_clusters = clusters;

	// initialize layers
	this->_clustering = clustering;
}

/**
 * Destruct a model.
 */
ClusteringModel::~ClusteringModel()
{
	delete this->_clustering;
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

	// get data matrix X
	Matrix X = input.load_data();

	// perform clustering on X
	std::vector<std::vector<int>> models;

	for ( int k : this->_clusters ) {
		std::vector<int> model = this->_clustering->compute(X, k);

		models.push_back(model);
	}

	// select the best model
	std::vector<int> Y_pred = models[0];

	log(LL_VERBOSE, "");

	return Y_pred;
}

}
