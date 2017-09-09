/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef CLUSTERINGMODEL_H
#define CLUSTERINGMODEL_H

#include "clustering/clustering.h"
#include "criterion/criterion.h"
#include "data/dataset.h"
#include "feature/feature.h"

namespace ML {

class ClusteringModel {
private:
	// input data
	Dataset _input;

	// feature layer
	FeatureLayer *_feature;
	Matrix _P;

	// clustering, criterion layers
	std::vector<ClusteringLayer *> _layers;
	CriterionLayer *_criterion;

public:
	ClusteringModel(FeatureLayer *feature, const std::vector<ClusteringLayer *>& layers, CriterionLayer *criterion);
	~ClusteringModel() {};

	void extract(const Dataset& input);
	std::vector<int> predict();

	void print_results(const std::vector<int>& Y_pred) const;
};

}

#endif
