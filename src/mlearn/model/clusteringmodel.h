/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef ML_MODEL_CLUSTERINGMODEL_H
#define ML_MODEL_CLUSTERINGMODEL_H

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
	std::vector<int> predict(const std::vector<Matrix>& X);
	void score(const Dataset& dataset, const std::vector<int>& y_pred);

	void print_results(const Dataset& dataset, const std::vector<int>& y_pred) const;
	void print_stats() const;

private:
	// layers
	std::vector<ClusteringLayer *> _clustering;
	CriterionLayer *_criterion;
	ClusteringLayer *_best_layer;

	// performance, accuracy stats
	struct {
		float error_rate;
		float fit_time;
		float predict_time;
	} _stats;
};



}

#endif
