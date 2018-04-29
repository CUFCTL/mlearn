/**
 * @file clustering/kmeans.h
 *
 * Interface definitions for the k-means clustering layer.
 */
#ifndef KMEANS_H
#define KMEANS_H

#include "mlearn/clustering/clustering.h"



namespace ML {



class KMeansLayer : public ClusteringLayer {
public:
	KMeansLayer(int K);

	void fit(const std::vector<Matrix>& X);

	float entropy() const { return 0; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _K; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	const std::vector<int>& output() const { return _labels; }
	bool success() const { return true; }

	void print() const;

private:
	int _K;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _labels;
};



}

#endif
