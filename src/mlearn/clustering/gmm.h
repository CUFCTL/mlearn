/**
 * @file clustering/gmm.h
 *
 * Interface definitions for the Gaussian mixture model layer.
 */
#ifndef GMM_H
#define GMM_H

#include "mlearn/clustering/clustering.h"
#include "mlearn/clustering/parameterset.h"



namespace ML {



class GMMLayer : public ClusteringLayer {
public:
	GMMLayer(int k);

	int fit(const std::vector<Matrix>& X);

	float entropy() const { return _entropy; }
	float log_likelihood() const { return _log_likelihood; }
	int num_clusters() const { return _k; }
	int num_parameters() const { return _num_parameters; }
	int num_samples() const { return _num_samples; }
	const std::vector<int>& output() const { return _output; }

	void print() const;

private:
	ParameterSet initialize(const std::vector<Matrix>& X, int num_init, bool small_em);
	void E_step(const std::vector<Matrix>& X, const ParameterSet& theta, Matrix& c);
	void M_step(const std::vector<Matrix>& X, const Matrix& c, ParameterSet& theta);

	int _k;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;
};



}

#endif
