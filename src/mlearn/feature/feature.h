/**
 * @file feature/feature.h
 *
 * Interface definitions for the abstract feature layer.
 */
#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include "mlearn/layer/layer.h"
#include "mlearn/math/matrix.h"



namespace ML {



class FeatureLayer : public Layer {
public:
	virtual ~FeatureLayer() {};

	virtual void fit(const Matrix& X, const std::vector<int>& y, int c) = 0;
	virtual Matrix transform(const Matrix& X) = 0;
};



}

#endif
