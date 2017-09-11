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
 * Compute the value of a Gaussian distribution at x:
 *
 *   h(x | mu, S) = (2pi)^(d/2) |S|^-0.5 e^(-1/2 (x - mu)' S^-1 (x - mu))
 *
 * @param x
 * @param mu
 * @param S_det
 * @param S_inv
 */
precision_t pdf(Matrix x, const Matrix& mu, precision_t S_det, const Matrix& S_inv)
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
Matrix ParameterSet::pdf_all(const Matrix& X) const
{
	int n = X.cols();
	Matrix h(n, this->_k);

	for ( int j = 0; j < this->_k; j++ ) {
		precision_t S_det = this->_S[j].determinant();
		Matrix S_inv = this->_S[j].inverse();

		for ( int i = 0; i < n; i++ ) {
			h.elem(i, j) = pdf(X(i), this->_mu[j], S_det, S_inv);
		}
	}

	return h;
}

/**
 * Compute the log-likelihood of a parameter set:
 *
 *   L = sum(log(sum(p_j * h_ij, j=1:k)), i=1:n)
 *
 * @param X
 */
precision_t ParameterSet::log_likelihood(const Matrix& X) const
{
	int n = X.cols();
	Matrix h = this->pdf_all(X);

	// compute L = sum(L_i, i=1:n)
	precision_t L = 0;

	for ( int i = 0; i < n; i++ ) {
		// compute L_i = log(sum(p_j * h_ij, j=1:k))
		precision_t sum = 0;
		for ( int j = 0; j < this->_k; j++ ) {
			sum += this->_p[j] * h.elem(i, j);
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

	// compute conditional probabilities (using nearest neighbor)
	Matrix c = Matrix::zeros(n, this->_k);

	for ( int i = 0; i < n; i++ ) {
		int min_j = -1;
		precision_t min_dist;

		for ( int j = 0; j < this->_k; j++ ) {
			precision_t dist = m_dist_L2(X, i, this->_mu[j], 0);

			if ( min_j == -1 || dist < min_dist ) {
				min_j = j;
				min_dist = dist;
			}
		}

		c.elem(i, min_j) = 1;
	}

	// update mixture proportions, covariances
	for ( int j = 0; j < this->_k; j++ ) {
		// compute n_j = sum(c_ij, i=1:n)
		precision_t n_j = 0;
		for ( int i = 0; i < n; i++ ) {
			n_j += c.elem(i, j);
		}

		// compute p_j = sum(c_ij, i=1:n) / n
		this->_p.push_back(n_j / n);

		// compute S_j = W_j / n_j
		Matrix W_j = Matrix::zeros(d, d);

		for ( int i = 0; i < n; i++ ) {
			if ( c.elem(i, j) > 0 ) {
				Matrix x_i = X(i) - this->_mu[j];
				Matrix W_ji = x_i * x_i.T();

				W_ji *= c.elem(i, j);
				W_j += W_ji;
			}
		}

		W_j /= n_j;

		this->_S.push_back(W_j);
	}
}

}
