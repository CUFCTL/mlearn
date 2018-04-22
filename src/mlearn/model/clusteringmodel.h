/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef CLUSTERINGMODEL_H
#define CLUSTERINGMODEL_H

#include "mlearn/clustering/clustering.h"
#include "mlearn/criterion/criterion.h"
#include "mlearn/data/dataset.h"



namespace ML {



class ClusteringModel {
public:
	ClusteringModel(const std::vector<ClusteringLayer *>& clustering, CriterionLayer *criterion);
	~ClusteringModel() {}

	void print() const;

	ClusteringLayer * best_layer() const { return _best_layer; }

	void fit(const std::vector<Matrix>& X);
	void score(const Dataset& input, const std::vector<int>& y_pred);

	void print_results(const Dataset& input, const std::vector<int>& y_pred) const;
	void print_stats() const;

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
};



}

#endif
