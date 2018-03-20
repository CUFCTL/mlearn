/**
 * @file clustering/parameterset.h
 *
 * Interface definitions for the parameter set.
 */
#ifndef PARAMETERSET_H
#define PARAMETERSET_H

#include <vector>
#include "mlearn/math/matrix.h"



namespace ML {



class ParameterSet {
public:
	// constructor/destructor functions
	ParameterSet(int k);
	ParameterSet(ParameterSet&& theta);
	ParameterSet() {};
	~ParameterSet() {};

	// I/O functions
	void print() const;

	// getter functions
	const float& n(int i) const { return _n[i]; }
	const float& p(int i) const { return _p[i]; }
	const Matrix& mu(int i) const { return _mu[i]; }
	const Matrix& S(int i) const { return _S[i]; }

	float& n(int i) { return _n[i]; }
	float& p(int i) { return _p[i]; }
	Matrix& mu(int i) { return _mu[i]; }
	Matrix& S(int i) { return _S[i]; }

	const std::vector<std::vector<Matrix>>& Xsubs() const { return _Xsubs; }
	const Matrix& h() const { return _h; }

	float log_likelihood() const;

	// mutator functions
	void initialize(const std::vector<Matrix>& X);
	void subtract_means(const std::vector<Matrix>& X);
	void pdf_all();

	// operators
	inline ParameterSet& operator=(ParameterSet rhs) { swap(*this, rhs); return *this; }

	// friend functions
	friend void swap(ParameterSet& lhs, ParameterSet& rhs);

private:
	int _k;
	std::vector<float> _n;
	std::vector<float> _p;
	std::vector<Matrix> _mu;
	std::vector<Matrix> _S;

	std::vector<std::vector<Matrix>> _Xsubs;
	Matrix _h;
};



}

#endif
