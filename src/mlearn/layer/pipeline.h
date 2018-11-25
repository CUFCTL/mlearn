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



class Pipeline : public EstimatorLayer {
public:
	Pipeline(std::vector<TransformerLayer *> transforms, EstimatorLayer *estimator);
	~Pipeline() {}

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X) const;
	float score(const Matrix& X, const std::vector<int>& y) const;

private:
	std::vector<TransformerLayer *> _transforms;
	EstimatorLayer *_estimator;
};



}

#endif
