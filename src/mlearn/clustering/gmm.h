/**
 * @file clustering/gmm.h
 *
 * Interface definitions for the Gaussian mixture model layer.
 */
#ifndef GMM_H
#define GMM_H

#include "mlearn/clustering/clustering.h"



namespace ML {



class GMMLayer : public ClusteringLayer {
public:
	GMMLayer(int K);

	class Component {
	public:
		Component() = default;

		void initialize(float pi, const Matrix& mu);
		void prepare();
		void compute_log_mv_norm(const std::vector<Matrix>& X, float *logP);

		float pi;
		Matrix mu;
		Matrix sigma;

	private:
		Matrix _sigma_inv;
		float _normalizer;
	};

	void fit(const std::vector<Matrix>& X);

	float entropy() const { return _entropy; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _K; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	const std::vector<int>& output() const { return _labels; }
	bool success() const { return _success; }

	void print() const;

private:
	void kmeans(const std::vector<Matrix>& X);
	void compute_log_mv_norm(const std::vector<Matrix>& X, float *loggamma);
	void compute_log_likelihood_gamma_nk(const float *logpi, int K, float *loggamma, int N, float *logL);
	void compute_log_gamma_k(const float *loggamma, int N, int K, float *logGamma);
	float compute_log_gamma_sum(const float *logpi, int K, const float *logGamma);
	void update(float *logpi, int K, float *loggamma, float *logGamma, float logGammaSum, const std::vector<Matrix>& X);
	std::vector<int> compute_labels(float *loggamma, int N, int K);
	float compute_entropy(float *loggamma, int N, const std::vector<int>& labels);

	int _K;
	std::vector<Component> _components;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _labels;
	bool _success;
};



}

#endif
