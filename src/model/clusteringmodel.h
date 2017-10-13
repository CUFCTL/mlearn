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
	// clustering layers
	std::vector<ClusteringLayer *> _clustering;

	// criterion layer
	CriterionLayer *_criterion;

	// selected clustering layer
	ClusteringLayer *_best_layer;

	// performance, accuracy stats
	struct {
		float error_rate;
		float predict_time;
	} _stats;

public:
	ClusteringModel(const std::vector<ClusteringLayer *>& clustering, CriterionLayer *criterion);
	~ClusteringModel() {};

	ClusteringLayer * best_layer() const { return this->_best_layer; };

	void predict(const std::vector<Matrix>& X);
	void validate(const Dataset& input, const std::vector<int>& Y_pred);

	void print_results(const Dataset& input, const std::vector<int>& Y_pred) const;
	void print_stats() const;
};

}

#endif
