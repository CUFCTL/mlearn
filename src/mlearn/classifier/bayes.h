/**
 * @file classifier/bayes.h
 *
 * Interface definitions for the naive Bayes classifier.
 */
#ifndef MLEARN_CLASSIFIER_BAYES_H
#define MLEARN_CLASSIFIER_BAYES_H

#include "mlearn/layer/estimator.h"



namespace mlearn {



class BayesLayer : public EstimatorLayer {
public:
	BayesLayer() = default;

	void fit(const Matrix& X) {}
	void fit(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

private:
	float prob(Matrix x, const Matrix& mu, const Matrix& S_inv) const;

	std::vector<Matrix> _mu;
	std::vector<Matrix> _S_inv;
};



}

#endif
