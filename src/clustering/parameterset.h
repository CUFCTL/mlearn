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
	std::vector<float> _p;
	std::vector<Matrix> _mu;
	std::vector<Matrix> _S;
	Matrix _h;

public:
	// constructor/destructor functions
	ParameterSet(int k);
	ParameterSet(ParameterSet&& theta);
	ParameterSet() {};
	~ParameterSet() {};

	// I/O functions
	void print() const;

	// getter functions
	const std::vector<float>& p() const { return this->_p; }
	const std::vector<Matrix>& mu() const { return this->_mu; }
	const std::vector<Matrix>& S() const { return this->_S; }
	const Matrix& h() const { return this->_h; }

	const float& p(int i) const { return this->_p[i]; }
	const Matrix& mu(int i) const { return this->_mu[i]; }
	const Matrix& S(int i) const { return this->_S[i]; }

	float& p(int i) { return this->_p[i]; }
	Matrix& mu(int i) { return this->_mu[i]; }
	Matrix& S(int i) { return this->_S[i]; }

	float log_likelihood(const std::vector<Matrix>& X) const;

	// mutator functions
	void initialize(const std::vector<Matrix>& X);
	void pdf_all(const std::vector<Matrix>& X);

	// operators
	inline ParameterSet& operator=(ParameterSet rhs) { swap(*this, rhs); return *this; }

	// friend functions
	friend void swap(ParameterSet& lhs, ParameterSet& rhs);
};

}

#endif
