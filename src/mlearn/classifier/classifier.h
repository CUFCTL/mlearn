/**
 * @file classifier/classifier.h
 *
 * Interface definitions for the abstract classifier layer.
 */
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include "mlearn/math/matrix.h"
#include "mlearn/util/iodevice.h"



namespace ML {



class ClassifierLayer : public IODevice {
public:
	virtual ~ClassifierLayer() {};
	virtual void compute(const Matrix& X, const std::vector<int>& y, int c) = 0;
	virtual std::vector<int> predict(const Matrix& X_test) = 0;
};



}

#endif
