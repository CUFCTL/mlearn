/**
 * @file clustering/gmm.h
 *
 * Interface definitions for the Gaussian mixture model layer.
 */
#ifndef GMM_H
#define GMM_H

#include "clustering/clustering.h"

namespace ML {

typedef struct {
	std::vector<precision_t> p;
	std::vector<Matrix> mu;
	std::vector<Matrix> S;
} parameter_t;

class GMMLayer : public ClusteringLayer {
private:
	int _k;
	precision_t _log_likelihood;
	int _num_parameters;
	int _num_samples;
	std::vector<int> _output;

	Matrix pdf(const Matrix& X, const parameter_t& theta);
	precision_t log_likelihood(const Matrix& X, const parameter_t& theta);
	parameter_t init_random(const Matrix& X, int num_init);
	void E_step(const Matrix& X, const parameter_t& theta, Matrix& c);
	void M_step(const Matrix& X, const Matrix& c, parameter_t& theta);

public:
	GMMLayer(int k);

	void compute(const Matrix& X);

	precision_t log_likelihood() const { return this->_log_likelihood; };
	int num_parameters() const { return this->_num_parameters; };
	int num_samples() const { return this->_num_samples; };
	inline std::vector<int> output() const { return this->_output; };

	void print() const;
};

}

#endif
