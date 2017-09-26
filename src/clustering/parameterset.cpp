/**
 * @file clustering/parameterset.cpp
 *
 * Implementation of the parameter set.
 */
#include <cmath>
#include "clustering/parameterset.h"
#include "math/matrix_utils.h"

namespace ML {

/**
 * Construct a parameter set.
 *
 * @param k
 */
ParameterSet::ParameterSet(int k)
{
	this->_k = k;
}

/**
 * Move-construct a parameter set.
 *
 * @param theta
 */
ParameterSet::ParameterSet(ParameterSet&& theta)
	: ParameterSet()
{
	swap(*this, theta);
}

/**
 * Print a parameter set.
 */
void ParameterSet::print() const
{
	std::cout << "k = " << this->_k << "\n";

	for ( int i = 0; i < this->_k; i++ ) {
		std::cout << "p_" << i + 1 << " = " << this->_p[i] << "\n";

		std::cout << "mu_" << i + 1 << " =\n";
		this->_mu[i].print(std::cout);

		std::cout << "S_" << i + 1 << " =\n";
		this->_S[i].print(std::cout);
	}

	std::cout << "\n";
}

/**
 * Compute the value of a Gaussian distribution at x:
 *
 *   h(x | mu, S) = (2pi)^(d/2) |S|^-0.5 e^(-1/2 (x - mu)' S^-1 (x - mu))
 *
 * @param x
 * @param mu
 * @param S_det
 * @param S_inv
 */
float pdf(Matrix x, const Matrix& mu, float S_det, const Matrix& S_inv)
{
	x -= mu;

	return powf(2 * M_PI, x.rows() / 2.0f) * powf(S_det, -0.5f) * expf(-0.5f * (x.T() * S_inv).dot(x));
}

/**
 * Compute h_ij for all i, j:
 *
 *   h_ij = h(x_i | mu_j, S_j)
 *
 * @param X
 */
void ParameterSet::pdf_all(const Matrix& X)
{
	int n = X.cols();

	for ( int j = 0; j < this->_k; j++ ) {
		float S_det = this->_S[j].determinant();
		Matrix S_inv = this->_S[j].inverse();

		for ( int i = 0; i < n; i++ ) {
			this->_h.elem(i, j) = pdf(X(i), this->_mu[j], S_det, S_inv);
		}
	}
}

/**
 * Compute the log-likelihood of a parameter set:
 *
 *   L = sum(log(sum(p_j * h_ij, j=1:k)), i=1:n)
 *
 * @param X
 */
float ParameterSet::log_likelihood(const Matrix& X) const
{
	int n = X.cols();

	// compute L = sum(L_i, i=1:n)
	float L = 0;

	for ( int i = 0; i < n; i++ ) {
		// compute L_i = log(sum(p_j * h_ij, j=1:k))
		float sum = 0;
		for ( int j = 0; j < this->_k; j++ ) {
			sum += this->_p[j] * this->_h.elem(i, j);
		}

		L += logf(sum);
	}

	return L;
}

/**
 * Initialize a parameter set by selecting k means randomly.
 *
 * @param X
 */
void ParameterSet::initialize(const Matrix& X)
{
	int n = X.cols();
	int d = X.rows();

	// choose k means randomly from data
	this->_mu = m_random_sample(X, this->_k);

	// initialize mixture proportions, covariances
	this->_p.reserve(this->_k);
	this->_S.reserve(this->_k);

	for ( int j = 0; j < this->_k; j++ ) {
		// initialize p_j = 1 / k
		this->_p.push_back(1.0f / this->_k);

		// initialize S_j = I_d
		this->_S.push_back(Matrix::identity(d));
	}

	// initialize h
	this->_h = Matrix(n, this->_k);
	this->pdf_all(X);
}

/**
 * Swap function for ParameterSet.
 *
 * @param lhs
 * @param rhs
 */
void swap(ParameterSet& lhs, ParameterSet& rhs)
{
	std::swap(lhs._k, rhs._k);
	std::swap(lhs._p, rhs._p);
	std::swap(lhs._mu, rhs._mu);
	std::swap(lhs._S, rhs._S);
	std::swap(lhs._h, rhs._h);
}

}
