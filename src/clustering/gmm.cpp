/**
 * @file clustering/gmm.cpp
 *
 * Implementation of Gaussian mixture models.
 */
#include <cmath>
#include <cstdlib>
#include "clustering/gmm.h"
#include "math/matrix_utils.h"
#include "util/logger.h"
#include "util/timer.h"

namespace ML {

const int INIT_SMALL_EM = false;
const int INIT_NUM_ITER = 5;
const precision_t INIT_EPSILON = 1e-2;

const int NUM_INIT = 10;
const int NUM_ITER = 200;
const precision_t EPSILON = 1e-3;

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
	Matrix temp = TRAN(x) * S_inv * x;

	return powf(2 * M_PI, x.rows() / 2.0f) * powf(S_det, -0.5f) * expf(-0.5f * temp.elem(0, 0));
}

/**
 * Compute h_ij for all i,j:
 *
 *   h_ij = h(x_i | mu_j, S_j)
 *
 * @param X
 * @param theta
 */
Matrix GMMLayer::pdf_all(const Matrix& X, const parameter_t& theta)
{
	int n = X.cols();
	Matrix h(n, this->_k);

	for ( int j = 0; j < this->_k; j++ ) {
		precision_t S_det = theta.S[j].determinant();
		Matrix S_inv = theta.S[j].inverse();

		for ( int i = 0; i < n; i++ ) {
			h.elem(i, j) = pdf(X(i), theta.mu[j], S_det, S_inv);
		}
	}

	return h;
}

/**
 * Compute the log-likelihood of a model:
 *
 *   L = sum(log(sum(p_j * h_ij, j=1:k)), i=1:n)
 *
 * @param X
 * @param theta
 */
precision_t GMMLayer::log_likelihood(const Matrix& X, const parameter_t& theta)
{
	int n = X.cols();
	Matrix h = this->pdf_all(X, theta);

	// compute L = sum(L_i, i=1:n)
	precision_t L = 0;

	for ( int i = 0; i < n; i++ ) {
		// compute L_i = log(sum(p_j * h_ij, j=1:k))
		precision_t sum = 0;
		for ( int j = 0; j < this->_k; j++ ) {
			sum += theta.p[j] * h.elem(i, j);
		}

		L += logf(sum);
	}

	return L;
}

/**
 * Initialize a random parameter set by selecting k means randomly.
 *
 * @param X
 * @param k
 */
parameter_t random_parameter(const Matrix& X, int k)
{
	int n = X.cols();
	int d = X.rows();
	parameter_t theta;

	// choose k means randomly from data
	for ( int j = 0; j < k; j++ ) {
		int i = lrand48() % n;
		theta.mu.push_back(X(i));
	}

	// compute conditional probabilities (using nearest neighbor)
	Matrix c = Matrix::zeros(n, k);

	for ( int i = 0; i < n; i++ ) {
		int min_j = -1;
		precision_t min_dist;

		for ( int j = 0; j < k; j++ ) {
			precision_t dist = m_dist_L2(X, i, theta.mu[j], 0);

			if ( min_j == -1 || dist < min_dist ) {
				min_j = j;
				min_dist = dist;
			}
		}

		c.elem(i, min_j) = 1;
	}

	// update mixture proportions, covariances
	for ( int j = 0; j < k; j++ ) {
		// compute n_j = sum(c_ij, i=1:n)
		precision_t n_j = 0;
		for ( int i = 0; i < n; i++ ) {
			n_j += c.elem(i, j);
		}

		// compute p_j = sum(c_ij, i=1:n) / n
		theta.p.push_back(n_j / n);

		// compute S_j = W_j / n_j
		Matrix W_j = Matrix::zeros(d, d);

		for ( int i = 0; i < n; i++ ) {
			if ( c.elem(i, j) > 0 ) {
				Matrix x_i = X(i) - theta.mu[j];
				Matrix W_ji = x_i * TRAN(x_i);

				W_ji *= c.elem(i, j);
				W_j += W_ji;
			}
		}

		W_j /= n_j;

		theta.S.push_back(W_j);
	}

	return theta;
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
parameter_t GMMLayer::initialize(const Matrix& X, int num_init, bool small_em)
{
	int n = X.cols();
	parameter_t theta;
	precision_t L_theta = 0;

	for ( int t = 0; t < num_init; t++ ) {
		parameter_t theta_test = random_parameter(X, this->_k);

		if ( small_em ) {
			// run the EM algorithm briefly
			Matrix c(n, this->_k);
			precision_t L0 = 0;

			for ( int m = 0; m < INIT_NUM_ITER; m++ ) {
				this->E_step(X, theta, c);
				this->M_step(X, c, theta);

				precision_t L1 = this->log_likelihood(X, theta);

				if ( L0 != 0 && fabsf(L1 - L0) < INIT_EPSILON ) {
					break;
				}

				L0 = L1;
			}
		}

		// keep theta_test if log-likelihood is greater
		precision_t L_test = this->log_likelihood(X, theta_test);

		if ( L_theta == 0 || L_theta < L_test ) {
			theta = theta_test;
			L_theta = L_test;
		}
	}

	return theta;
}

/**
 * Compute the conditional probability that z_ik = 1
 * for all i,j:
 *
 *   c_ij = p_j * h_ij / sum(p_l * h_il, l=1:k)
 *
 * @param X
 * @param theta
 * @param c
 */
void GMMLayer::E_step(const Matrix& X, const parameter_t& theta, Matrix& c)
{
	// compute h_ij for each i,j
	int n = X.cols();
	Matrix h = this->pdf_all(X, theta);

	// compute c_ij for each i,j
	for ( int i = 0; i < n; i++ ) {
		precision_t sum = 0;

		for ( int j = 0; j < this->_k; j++ ) {
			sum += theta.p[j] * h.elem(i, j);
		}

		for ( int j = 0; j < this->_k; j++ ) {
			c.elem(i, j) = theta.p[j] * h.elem(i, j) / sum;
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
void GMMLayer::M_step(const Matrix& X, const Matrix& c, parameter_t& theta)
{
	int n = X.cols();
	int d = X.rows();

	for ( int j = 0; j < this->_k; j++ ) {
		// compute n_j = sum(c_ij, i=1:n)
		precision_t n_j = 0;
		for ( int i = 0; i < n; i++ ) {
			n_j += c.elem(i, j);
		}

		// compute p_j = sum(c_ij, i=1:n) / n
		theta.p[j] = n_j / n;

		// compute mu_j = sum(c_ij * x_i, i=1:n) / n_j
		Matrix sum = Matrix::zeros(d, 1);
		for ( int i = 0; i < n; i++ ) {
			sum += c.elem(i, j) * X(i);
		}

		theta.mu[j] = sum / n_j;

		// compute S_j = W_j / n_j
		Matrix W_j = Matrix::zeros(d, d);

		for ( int i = 0; i < n; i++ ) {
			Matrix x_i = X(i) - theta.mu[j];
			Matrix W_ji = x_i * TRAN(x_i);

			W_ji *= c.elem(i, j);
			W_j += W_ji;
		}

		W_j /= n_j;

		theta.S[j] = W_j;
	}
}

/**
 * Partition a matrix X of observations into
 * clusters using a Gaussian mixture model.
 *
 * @param X
 */
void GMMLayer::compute(const Matrix& X)
{
	int n = X.cols();
	int d = X.rows();

	timer_push("Gaussian mixture model");

	timer_push("initialize parameters");

	parameter_t theta = this->initialize(X, NUM_INIT, INIT_SMALL_EM);

	timer_pop();

	timer_push("run EM algorithm");

	Matrix c(n, this->_k);
	precision_t L0 = 0;

	for ( int m = 0; m < NUM_ITER; m++ ) {
		timer_push("E step");

		this->E_step(X, theta, c);

		timer_pop();

		timer_push("M step");

		this->M_step(X, c, theta);

		timer_pop();

		timer_push("check for convergence");

		precision_t L1 = this->log_likelihood(X, theta);

		if ( L0 != 0 && fabsf(L1 - L0) < EPSILON ) {
			log(LL_DEBUG, "converged after %d iteratinos", m + 1);
			break;
		}

		L0 = L1;

		timer_pop();
	}

	timer_pop();

	timer_push("assign labels");

	std::vector<int> y;

	for ( int i = 0; i < n; i++ ) {
		int max_j = -1;
		precision_t max_c;

		for ( int j = 0; j < this->_k; j++ ) {
			if ( max_j == -1 || max_c < c.elem(i, j) ) {
				max_j = j;
				max_c = c.elem(i, j);
			}
		}

		y.push_back(max_j);
	}

	timer_pop();

	// save outputs
	this->_log_likelihood = this->log_likelihood(X, theta);
	this->_num_parameters = this->_k * d + this->_k;
	this->_num_samples = n;
	this->_output = y;

	timer_pop();
}

/**
 * Print a GMM layer.
 */
void GMMLayer::print() const
{
	log(LL_INFO, "Gaussian mixture model");
	log(LL_INFO, "  k: %d", this->_k);
	log(LL_INFO, "");
}

}
