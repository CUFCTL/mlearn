/**
 * @file model/clusteringmodel.h
 *
 * Interface definitions for the clustering model.
 */
#ifndef MLEARN_MODEL_CLUSTERINGMODEL_H
#define MLEARN_MODEL_CLUSTERINGMODEL_H

#include "mlearn/clustering/clustering.h"
#include "mlearn/criterion/criterion.h"
#include "mlearn/data/dataset.h"



namespace mlearn {



class ClusteringModel {
public:
	ClusteringModel(const std::vector<ClusteringLayer *>& models, CriterionLayer *criterion, ClusteringLayer *clustering=nullptr);
	~ClusteringModel() {}

	void save(const std::string& path);
	void load(const std::string& path);
	void print() const;

	void fit(const std::vector<Matrix>& X);
	std::vector<int> predict(const std::vector<Matrix>& X);
	void score(const Dataset& dataset, const std::vector<int>& y_pred);

	void print_results(const Dataset& dataset, const std::vector<int>& y_pred) const;
	void print_stats() const;

private:
	// layers
	std::vector<ClusteringLayer *> _models;
	CriterionLayer *_criterion;
	ClusteringLayer *_clustering;

	// performance, accuracy stats
	struct {
		float error_rate;
		float fit_time;
		float predict_time;
	} _stats;
};



}

#endif
