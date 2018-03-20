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
public:
	KMeansLayer(int k);

	int fit(const std::vector<Matrix>& X);

	float entropy() const { return _entropy; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _k; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	const std::vector<int>& output() const { return _output; }

	void print() const;

private:
	void E_step(const std::vector<Matrix>& X, const ParameterSet& theta, std::vector<int>& y);
	void M_step(const std::vector<Matrix>& X, const std::vector<int>& y, ParameterSet& theta);

	int _k;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;
};



}

#endif
