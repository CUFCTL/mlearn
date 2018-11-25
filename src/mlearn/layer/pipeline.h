/**
 * @file layer/pipeline.h
 *
 * Interface definitions for the pipeline.
 */
#ifndef MLEARN_LAYER_PIPELINE_H
#define MLEARN_LAYER_PIPELINE_H

#include "mlearn/layer/estimator.h"
#include "mlearn/layer/transformer.h"



namespace mlearn {



class Pipeline {
public:
	Pipeline(std::vector<TransformerLayer *> transforms, EstimatorLayer *estimator);
	~Pipeline() {}

	void save(const std::string& path);
	void load(const std::string& path);
	void print() const;

	void fit(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X) const;
	float score(const Matrix& X, const std::vector<int>& y) const;

private:
	std::vector<TransformerLayer *> _transforms;
	EstimatorLayer *_estimator;
};



}

#endif
