/**
 * @file clustering/gmm.h
 *
 * Interface definitions for the Gaussian mixture model layer.
 */
#ifndef GMM_H
#define GMM_H

#include "clustering/clustering.h"
#include "clustering/parameterset.h"

namespace ML {

class GMMLayer : public ClusteringLayer {
private:
	int _k;
	float _entropy;
	float _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;

	ParameterSet initialize(const Matrix& X, int num_init, bool small_em);
	void E_step(const Matrix& X, const ParameterSet& theta, Matrix& c);
	void M_step(const Matrix& X, const Matrix& c, ParameterSet& theta);

public:
	GMMLayer(int k);

	void compute(const Matrix& X);

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
