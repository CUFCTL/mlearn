/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef CLUSTERINGMODEL_H
#define CLUSTERINGMODEL_H

#include "clustering/clustering.h"
#include "clustering/criterion.h"
#include "data/dataset.h"

namespace ML {

class ClusteringModel {
private:
	// layers
	// InitLayer *_init;
	std::vector<ClusteringLayer *> _layers;
	CriterionLayer *_criterion;

public:
	ClusteringModel(const std::vector<ClusteringLayer *>& layers, CriterionLayer *criterion);
	~ClusteringModel() {};

	std::vector<int> run(const Dataset& input);
};

}

#endif
