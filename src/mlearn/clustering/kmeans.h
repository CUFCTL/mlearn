/**
 * @file clustering/kmeans.h
 *
 * Interface definitions for the k-means clustering layer.
 */
#ifndef MLEARN_CLUSTERING_KMEANS_H
#define MLEARN_CLUSTERING_KMEANS_H

#include "mlearn/clustering/clustering.h"



namespace mlearn {



class KMeansLayer : public ClusteringLayer {
public:
	KMeansLayer(int K);

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c) { fit(X); }
	std::vector<int> predict(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

	float entropy() const { return 0; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _K; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	bool success() const { return true; }

private:
	int _K;
	std::vector<Matrix> _means;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
};



}

#endif
