/**
 * @file clustering/gmm.cpp
 *
 * Implementation of Gaussian mixture models.
 */
#include <cmath>
#include <stdexcept>
#include "clustering/gmm.h"
#include "util/logger.h"
#include "util/timer.h"

namespace ML {

const int INIT_SMALL_EM = false;
const int INIT_NUM_ITER = 5;
const float INIT_EPSILON = 1e-2;

const int NUM_INIT = 10;
const int NUM_ITER = 200;
const float EPSILON = 1e-3;

/**
 * Construct a GMM layer.
 *
 * @param k
 */
GMMLayer::GMMLayer(int k)
{
	this->_k = k;
}

/**
 * Initialize a parameter set by selecting means randomly
 * from the data. The best set is selected from several such
 * sets by comparing the log-likelihood.
 *
 * If specified, the EM algorithm is run briefly on each
 * parameter set before evaluating its log-likelihood.
 *
 * @param X
 * @param num_init
 * @param small_em
 */
ParameterSet GMMLayer::initialize(const std::vector<Matrix>& X, int num_init, bool small_em)
{
	int n = X.size();
	ParameterSet theta(this->_k);
	float L_theta = 0;

	for ( int t = 0; t < num_init; t++ ) {
		ParameterSet theta_test(this->_k);

		theta_test.initialize(X);

		if ( small_em ) {
			// run the EM algorithm briefly
			Matrix c(n, this->_k);
			float L0 = 0;

			for ( int m = 0; m < INIT_NUM_ITER; m++ ) {
				this->E_step(X, theta_test, c);
				this->M_step(X, c, theta_test);

				float L1 = theta_test.log_likelihood();

				if ( L0 != 0 && fabsf(L1 - L0) < INIT_EPSILON ) {
					break;
				}

				L0 = L1;
			}
		}

		// keep the parameter set with greater log-likelihood
		float L_test = theta_test.log_likelihood();

		if ( L_theta == 0 || L_theta < L_test ) {
			theta = std::move(theta_test);
			L_theta = L_test;
		}
	}

	return theta;
}

/**
 * Compute the conditional probability that z_ik = 1
 * for all i,j given theta:
 *
 *   c_ij = p_j * h_ij / sum(p_l * h_il, l=1:k)
 *
 * @param X
 * @param theta
 * @param c
 */
void GMMLayer::E_step(const std::vector<Matrix>& X, const ParameterSet& theta, Matrix& c)
{
	const Matrix& h = theta.h();
	int n = h.rows();

	// compute c_ij for each i,j
	for ( int i = 0; i < n; i++ ) {
		float sum = 0;

		for ( int j = 0; j < this->_k; j++ ) {
			sum += theta.p(j) * h.elem(i, j);
		}

		for ( int j = 0; j < this->_k; j++ ) {
			c.elem(i, j) = theta.p(j) * h.elem(i, j) / sum;
		}
	}
}

/**
 * Compute the maximum-likelihood estimate of theta
 * from a data matrix X and conditional probabilities c.
 *
 * @param X
 * @param c
 * @param theta
 */
void GMMLayer::M_step(const std::vector<Matrix>& X, const Matrix& c, ParameterSet& theta)
{
	int n = X.size();

	// compute n_j = sum(c_ij, i=1:n)
	for ( int j = 0; j < this->_k; j++ ) {
		float& n_j = theta.n(j);
		n_j = 0;

		for ( int i = 0; i < n; i++ ) {
			n_j += c.elem(i, j);
		}
	}

	// compute p_j = n_j / n
	for ( int j = 0; j < this->_k; j++ ) {
		theta.p(j) = theta.n(j) / n;
	}

	// compute mu_j = sum(c_ij * x_i, i=1:n) / n_j
	for ( int j = 0; j < this->_k; j++ ) {
		Matrix& mu_j = theta.mu(j);
		mu_j.init_zeros();

		for ( int i = 0; i < n; i++ ) {
			mu_j.axpy(c.elem(i, j) / theta.n(j), X[i]);
		}
	}

	// update mean-subtracted data array
	theta.subtract_means(X);

	// compute S_j = sum(c_ij * (x_i - mu_j) * (x_i - mu_j)', i=1:n) / n_j
	const auto& Xsubs = theta.Xsubs();

	for ( int j = 0; j < this->_k; j++ ) {
		Matrix& S_j = theta.S(j);
		S_j.init_zeros();

		for ( int i = 0; i < n; i++ ) {
			S_j.gemm(c.elem(i, j) / theta.n(j), Xsubs[j][i], Xsubs[j][i].T(), 1.0f);
		}
	}

	// update pdf matrix
	theta.pdf_all();
}

/**
 * Compute labels for a dataset from the conditional
 * probability matrix.
 *
 * @param c
 */
std::vector<int> compute_labels(const Matrix& c)
{
	int n = c.rows();
	int k = c.cols();
	std::vector<int> y;

	y.reserve(n);

	for ( int i = 0; i < n; i++ ) {
		int max_j = -1;
		float max_c;

		for ( int j = 0; j < k; j++ ) {
			if ( max_j == -1 || max_c < c.elem(i, j) ) {
				max_j = j;
				max_c = c.elem(i, j);
			}
		}

		y.push_back(max_j);
	}

	return y;
}

/**
 * Compute the entropy of a model:
 *
 *   E = sum(sum(z_ij * ln(c_ij), j=1:n), i=1:n)
 *
 * @param c
 * @param y
 */
float compute_entropy(const Matrix& c, const std::vector<int>& y)
{
	int n = c.rows();
	float E = 0;

	for ( int i = 0; i < n; i++ ) {
		E += logf(c.elem(i, y[i]));
	}

	return E;
}

/**
 * Partition a matrix X of observations into clusters
 * using a Gaussian mixture model. Returns 0 on success,
 * 1 otherwise.
 *
 * @param X
 */
int GMMLayer::compute(const std::vector<Matrix>& X)
{
	int status = 0;

	Timer::push("Gaussian mixture model");

	try {
		int n = X.size();
		int d = X[0].rows();

		// initialize parameters
		ParameterSet theta = this->initialize(X, NUM_INIT, INIT_SMALL_EM);

		// run EM algorithm
		Matrix c(n, this->_k);
		float L0 = 0;

		for ( int m = 0; m < NUM_ITER; m++ ) {
			this->E_step(X, theta, c);
			this->M_step(X, c, theta);

			// check for convergence
			float L1 = theta.log_likelihood();

			if ( L0 != 0 && fabsf(L1 - L0) < EPSILON ) {
				log(LL_DEBUG, "converged after %d iteratinos", m + 1);
				break;
			}

			L0 = L1;
		}

		// save outputs
		std::vector<int> y = compute_labels(c);

		this->_entropy = compute_entropy(c, y);
		this->_log_likelihood = theta.log_likelihood();
		this->_num_parameters = this->_k * (1 + d + d * d);
		this->_num_samples = n;
		this->_output = y;
	}
	catch ( std::runtime_error& e ) {
		status = 1;
	}

	Timer::pop();

	return status;
}

/**
 * Print a GMM layer.
 */
void GMMLayer::print() const
{
	log(LL_VERBOSE, "Gaussian mixture model");
	log(LL_VERBOSE, "  %-20s  %10d", "k", this->_k);
}

}
