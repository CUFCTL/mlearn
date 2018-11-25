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

	int num_clusters() const { return _K; }
	float aic() const;
	float bic() const;
	float icl() const { return bic(); }

private:
	int _K;
	std::vector<Matrix> _means;
	float _log_likelihood {-INFINITY};
	int _num_parameters {0};
	int _num_samples {0};
};



}

#endif
