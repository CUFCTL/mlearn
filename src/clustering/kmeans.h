/**
 * @file clustering/kmeans.h
 *
 * Interface definitions for the k-means clustering layer.
 */
#ifndef KMEANS_H
#define KMEANS_H

#include "clustering/clustering.h"

namespace ML {

class KMeansLayer : public ClusteringLayer {
private:
	int _k;
	std::vector<int> _output;
	precision_t _error;

public:
	KMeansLayer(int k);

	void compute(const Matrix& X);

	inline std::vector<int> output() const { return this->_output; };
	inline precision_t error() const { return this->_error; };

	void print() const;
};

}

#endif
