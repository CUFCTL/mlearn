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
	precision_t _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;

public:
	KMeansLayer(int k);

	void compute(const Matrix& X);

	precision_t log_likelihood() const { return this->_log_likelihood; };
	int num_parameters() const { return this->_num_parameters; };
	int num_samples() const { return this->_num_samples; };
	inline std::vector<int> output() const { return this->_output; };

	void print() const;
};

}

#endif
