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

namespace ML {

class ClusteringModel {
private:
	// input data
	Dataset _input;

	// clustering layers
	std::vector<ClusteringLayer *> _clustering;

	// criterion layer
	CriterionLayer *_criterion;

	// selected clustering layer
	ClusteringLayer *_min_c;

	// performance, accuracy stats
	struct {
		float error_rate;
		float predict_time;
	} _stats;

public:
	ClusteringModel(const std::vector<ClusteringLayer *>& clustering, CriterionLayer *criterion);
	~ClusteringModel() {};

	std::vector<int> predict(const Dataset& input);
	void validate(const std::vector<int>& Y_pred);

	void print_results(const std::vector<int>& Y_pred) const;
	void print_stats() const;
};

}

#endif
