/**
 * @file classifier/knn.h
 *
 * Interface definitions for the kNN classifier.
 */
#ifndef KNN_H
#define KNN_H

#include "mlearn/classifier/classifier.h"
#include "mlearn/math/matrix_utils.h"

namespace ML {

class KNNLayer : public ClassifierLayer {
private:
	int k;
	dist_func_t dist;

public:
	KNNLayer(int k, dist_func_t dist);
	KNNLayer() : KNNLayer(1, m_dist_L1) {};

	std::vector<DataLabel> predict(
		const Matrix& X,
		const std::vector<DataEntry>& Y,
		const std::vector<DataLabel>& C,
		const Matrix& X_test
	);

	void print();
};

}

#endif
