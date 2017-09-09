/**
 * @file clustering/parameterset.h
 *
 * Interface definitions for the parameter set.
 */
#ifndef PARAMETERSET_H
#define PARAMETERSET_H

#include <vector>
#include "math/matrix.h"

namespace ML {

class ParameterSet {
private:
	int _k;
	std::vector<precision_t> _p;
	std::vector<Matrix> _mu;
	std::vector<Matrix> _S;

public:
	ParameterSet(int k);

	// getter functions
	const precision_t& p(int i) const { return this->_p[i]; }
	const Matrix& mu(int i) const { return this->_mu[i]; }
	const Matrix& S(int i) const { return this->_S[i]; }

	precision_t& p(int i) { return this->_p[i]; }
	Matrix& mu(int i) { return this->_mu[i]; }
	Matrix& S(int i) { return this->_S[i]; }

	precision_t log_likelihood(const Matrix& X) const;
	Matrix pdf_all(const Matrix& X) const;

	// mutator functions
	void initialize(const Matrix& X);
};

}

#endif