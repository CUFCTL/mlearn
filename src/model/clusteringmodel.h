/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef CLUSTERINGMODEL_H
#define CLUSTERINGMODEL_H

#include "clustering/clustering.h"
#include "data/dataset.h"

namespace ML {

class ClusteringModel {
private:
	// hyperparameters
	std::vector<int> _clusters;

	// layers
	// InitLayer *_init;
	ClusteringLayer *_clustering;
	// CriterionLayer *_criterion;

public:
	ClusteringModel(const std::vector<int>& clusters, ClusteringLayer *clustering);
	~ClusteringModel();

	std::vector<int> run(const Dataset& input);
};

}

#endif
