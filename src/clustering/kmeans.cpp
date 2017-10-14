/**
 * @file clustering/kmeans.cpp
 *
 * Implementation of k-means clustering.
 */
#include <cmath>
#include "clustering/kmeans.h"
#include "math/matrix_utils.h"
#include "util/logger.h"
#include "util/timer.h"

namespace ML {

/**
 * Construct a k-means layer.
 *
 * @param k
 */
KMeansLayer::KMeansLayer(int k)
{
	this->_k = k;
}

/**
 * Compute the label of each sample given the
 * mean of each cluster:
 *
 *   y_i = argmin(j, ||x_i - mu_j||)
 *
 * @param X
 * @param theta
 * @param y
 */
void KMeansLayer::E_step(const std::vector<Matrix>& X, const ParameterSet& theta, std::vector<int>& y)
{
	int n = X.size();

	for ( int i = 0; i < n; i++ ) {
		int min_j = -1;
		float min_dist;

		for ( int j = 0; j < this->_k; j++ ) {
			float dist = m_dist_L2(X[i], 0, theta.mu(j), 0);

			if ( min_j == -1 || dist < min_dist ) {
				min_j = j;
				min_dist = dist;
			}
		}

		y[i] = min_j;
	}
}

/**
 * Compute the mean of each cluster given the
 * label of each sample.
 *
 * @param X
 * @param y
 * @param theta
 */
void KMeansLayer::M_step(const std::vector<Matrix>& X, const std::vector<int>& y, ParameterSet& theta)
{
	int n = X.size();

	for ( int j = 0; j < this->_k; j++ ) {
		// compute n_j, the number of samples in cluster j
		int n_j = 0;

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				n_j++;
			}
		}

		// compute mu_j = mean of all x_i in cluster j
		Matrix& mu_j = theta.mu(j);
		mu_j.init_zeros();

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				mu_j += X[i];
			}
		}
		mu_j /= n_j;
	}
}

/**
 * Partition a matrix X of observations into
 * clusters using k-means clustering.
 *
 * @param X
 */
int KMeansLayer::compute(const std::vector<Matrix>& X)
{
	Timer::push("K-means");

	int n = X.size();
	int d = X[0].rows();

	// initialize parameters
	ParameterSet theta(this->_k);
	theta.initialize(X);

	// run k-means algorithm
	std::vector<int> y;
	std::vector<int> y_next(n);

	while ( true ) {
		E_step(X, theta, y_next);

		if ( y == y_next ) {
			break;
		}

		y = y_next;

		M_step(X, y, theta);
	}

	// compute mean-subtracted data for each mean
	std::vector<std::vector<Matrix>> X_subs = m_subtract_means(X, theta.mu());

	// compute additional parameters
	for ( int j = 0; j < this->_k; j++ ) {
		// compute n_j, the number of samples in cluster j
		float n_j = 0;

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				n_j++;
			}
		}

		// compute p_j = n_j / n
		theta.p(j) = n_j / n;

		// compute S_j = sum(c_ij * (x_i - mu_j) * (x_i - mu_j)', i=1:n) / n_j
		Matrix& S_j = theta.S(j);
		S_j.init_zeros();

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				S_j.gemm(1.0f, X_subs[j][i], X_subs[j][i].T(), 1.0f);
			}
		}
		S_j /= n_j;
	}

	theta.pdf_all(X);

	// save outputs
	this->_entropy = 0;
	this->_log_likelihood = theta.log_likelihood(X);
	this->_num_parameters = this->_k * (1 + d + d * d);
	this->_num_samples = n;
	this->_output = y;

	Timer::pop();

	return 0;
}

/**
 * Print a k-means layer.
 */
void KMeansLayer::print() const
{
	log(LL_VERBOSE, "K-means");
	log(LL_VERBOSE, "  %-20s  %10d", "k", this->_k);
}

}
