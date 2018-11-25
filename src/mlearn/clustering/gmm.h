/**
 * @file clustering/gmm.h
 *
 * Interface definitions for the Gaussian mixture model layer.
 */
#ifndef MLEARN_CLUSTERING_GMM_H
#define MLEARN_CLUSTERING_GMM_H

#include "mlearn/clustering/clustering.h"



namespace mlearn {



class GMMLayer : public ClusteringLayer {
public:
	GMMLayer(int K);

	class Component {
	public:
		Component() = default;

		void initialize(float pi, const Matrix& mu);
		void prepare();
		void compute_log_prob(const Matrix& X, Matrix& logP, int k) const;

		friend IODevice& operator<<(IODevice& file, const Component& component);
		friend IODevice& operator>>(IODevice& file, Component& component);

		float pi;
		Matrix mu;
		Matrix sigma;

	private:
		Matrix _sigma_inv;
		float _normalizer;
	};

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c) { fit(X); }
	std::vector<int> predict(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

	int num_clusters() const { return _K; }
	float aic() const;
	float bic() const;
	float icl() const;

private:
	void kmeans(const Matrix& X);
	float e_step(const Matrix& X, Matrix& gamma) const;
	void m_step(const Matrix& X, const Matrix& gamma);
	std::vector<int> compute_labels(const Matrix& gamma) const;
	float compute_entropy(const Matrix& gamma, const std::vector<int>& labels) const;

	int _K;
	std::vector<Component> _components;
	float _entropy {0};
	float _log_likelihood {-INFINITY};
	int _num_parameters {0};
	int _num_samples {0};
};



}

#endif
