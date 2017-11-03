/**
 * @file clustering/kmeans.h
 *
 * Interface definitions for the k-means clustering layer.
 */
#ifndef KMEANS_H
#define KMEANS_H

#include "mlearn/clustering/clustering.h"
#include "mlearn/clustering/parameterset.h"

namespace ML {

class KMeansLayer : public ClusteringLayer {
private:
	int _k;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;

	void E_step(const std::vector<Matrix>& X, const ParameterSet& theta, std::vector<int>& y);
	void M_step(const std::vector<Matrix>& X, const std::vector<int>& y, ParameterSet& theta);

public:
	KMeansLayer(int k);

	int compute(const std::vector<Matrix>& X);

	float entropy() const { return this->_entropy; }
	float log_likelihood() const { return this->_log_likelihood; }
	int num_clusters() const { return this->_k; }
	int num_parameters() const { return this->_num_parameters; }
	int num_samples() const { return this->_num_samples; }
	inline std::vector<int> output() const { return this->_output; }

	void print() const;
};

}

#endif
