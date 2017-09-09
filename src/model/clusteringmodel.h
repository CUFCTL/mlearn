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
	Dataset _input;

	FeatureLayer *_feature;
	Matrix _P;

	std::vector<ClusteringLayer *> _layers;
	CriterionLayer *_criterion;

public:
	ClusteringModel(FeatureLayer *feature, const std::vector<ClusteringLayer *>& layers, CriterionLayer *criterion);
	~ClusteringModel() {};

	void extract(const Dataset& input);
	std::vector<int> predict();
};

}

#endif
