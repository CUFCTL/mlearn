/**
 * @file classifier.h
 *
 * Interface definitions for the abstract classifier layer.
 */
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include "data/dataset.h"
#include "math/matrix.h"

namespace ML {

class ClassifierLayer {
public:
	virtual ~ClassifierLayer() {};

	virtual std::vector<DataLabel> predict(
		const Matrix& X,
		const std::vector<DataEntry>& Y,
		const std::vector<DataLabel>& C,
		const Matrix& X_test
	) = 0;

	virtual void print() = 0;
};

}

#endif
