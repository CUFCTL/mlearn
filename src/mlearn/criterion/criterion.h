/**
 * @file criterion/criterion.h
 *
 * Interface definitions for the criterion layer.
 */
#ifndef MLEARN_CRITERION_CRITERION_H
#define MLEARN_CRITERION_CRITERION_H

#include <vector>
#include "mlearn/clustering/clustering.h"
#include "mlearn/layer/estimator.h"
#include "mlearn/util/iodevice.h"



namespace mlearn {



enum class Criterion {
	AIC,
	BIC,
	ICL
};



class CriterionLayer : public EstimatorLayer {
public:
	CriterionLayer(Criterion criterion, const std::vector<ClusteringLayer*>& models);
	virtual ~CriterionLayer() {}

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c) { fit(X); }
	std::vector<int> predict(const Matrix& X) const;
	float score(const Matrix& X, const std::vector<int>& y) const;

protected:
	Criterion _criterion;
	std::vector<ClusteringLayer*> _models;
	ClusteringLayer* _selected_model {nullptr};
};



}

#endif
