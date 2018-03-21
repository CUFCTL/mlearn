/**
 * @file classifier/knn.h
 *
 * Interface definitions for the kNN classifier.
 */
#ifndef KNN_H
#define KNN_H

#include "mlearn/classifier/classifier.h"



namespace ML {



enum class KNNDist {
	none,
	COS,
	L1,
	L2
};



class KNNLayer : public ClassifierLayer {
public:
	KNNLayer(int k, KNNDist dist);
	KNNLayer() : KNNLayer(1, KNNDist::L1) {}

	void compute(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X_test);

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

private:
	int _k;
	KNNDist _dist;
	Matrix _X;
	std::vector<int> _y;
};



}

#endif
