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
public:
	KNNLayer(int k, dist_func_t dist);
	KNNLayer() : KNNLayer(1, m_dist_L1) {};

	void compute(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X_test);
	void print();

private:
	int _k;
	dist_func_t _dist;
	Matrix _X;
	std::vector<int> _y;
};



}

#endif
