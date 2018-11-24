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
		void compute_log_prob(const std::vector<Matrix>& X, Matrix& logP, int k);

		friend IODevice& operator<<(IODevice& file, const Component& component);
		friend IODevice& operator>>(IODevice& file, Component& component);

		float pi;
		Matrix mu;
		Matrix sigma;

	private:
		Matrix _sigma_inv;
		float _normalizer;
	};

	void fit(const std::vector<Matrix>& X);
	std::vector<int> predict(const std::vector<Matrix>& X);

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

	float entropy() const { return _entropy; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _K; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	bool success() const { return _success; }

private:
	void kmeans(const std::vector<Matrix>& X);
	float e_step(const std::vector<Matrix>& X, Matrix& gamma);
	void m_step(const std::vector<Matrix>& X, const Matrix& gamma);
	std::vector<int> compute_labels(const Matrix& gamma);
	float compute_entropy(const Matrix& gamma, const std::vector<int>& labels);

	int _K;
	std::vector<Component> _components;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	bool _success;
};



}

#endif
