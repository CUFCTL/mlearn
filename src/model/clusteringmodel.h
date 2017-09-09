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

	// clustering layer
	ClusteringLayer *_clustering;

	// performance, accuracy stats
	struct {
		float error_rate;
		float extract_time;
		float predict_time;
	} _stats;

public:
	ClusteringModel(FeatureLayer *feature, ClusteringLayer *clustering);
	~ClusteringModel() {};

	void extract(const Dataset& input);
	std::vector<int> predict();
	void validate(const std::vector<int>& Y_pred);

	void print_results(const std::vector<int>& Y_pred) const;
	void print_stats() const;
};

}

#endif
