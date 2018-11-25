/**
 * @file layer/estimator.h
 *
 * Interface definitions for the abstract estimator layer.
 */
#ifndef MLEARN_LAYER_ESTIMATOR_H
#define MLEARN_LAYER_ESTIMATOR_H

#include <vector>
#include "mlearn/layer/layer.h"
#include "mlearn/math/matrix.h"



namespace mlearn {



class EstimatorLayer : public Layer {
public:
	virtual ~EstimatorLayer() {}

	virtual void fit(const Matrix& X) = 0;
	virtual void fit(const Matrix& X, const std::vector<int>& y, int c) = 0;
	virtual std::vector<int> predict(const Matrix& X) const = 0;
	virtual float score(const Matrix& X, const std::vector<int>& y) const;
};



}

#endif
