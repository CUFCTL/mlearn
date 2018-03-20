/**
 * @file feature/feature.h
 *
 * Interface definitions for the abstract feature layer.
 */
#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include "mlearn/math/matrix.h"
#include "mlearn/util/iodevice.h"



namespace ML {



class FeatureLayer : public IODevice {
public:
	virtual ~FeatureLayer() {};

	virtual void compute(const Matrix& X, const std::vector<int>& y, int c) = 0;
	virtual Matrix project(const Matrix& X) = 0;
};



}

#endif
