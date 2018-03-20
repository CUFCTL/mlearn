/**
 * @file clustering/kmeans.cpp
 *
 * Implementation of k-means clustering.
 */
#include <cmath>
#include <stdexcept>
#include "mlearn/clustering/kmeans.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"

namespace ML {

/**
 * Construct a k-means layer.
 *
 * @param k
 */
KMeansLayer::KMeansLayer(int k)
{
	_k = k;
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

		for ( int j = 0; j < _k; j++ ) {
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

	for ( int j = 0; j < _k; j++ ) {
		// compute n_j, the number of samples in cluster j
		float& n_j = theta.n(j);
		n_j = 0;

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
	int status = 0;

	Timer::push("K-means");

	try {
		int n = X.size();
		int d = X[0].rows();

		// initialize parameters
		ParameterSet theta(_k);
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

		// compute n_j, the number of samples in cluster j
		for ( int j = 0; j < _k; j++ ) {
			float& n_j = theta.n(j);
			n_j = 0;

			for ( int i = 0; i < n; i++ ) {
				if ( y[i] == j ) {
					n_j++;
				}
			}
		}

		// compute p_j = n_j / n
		for ( int j = 0; j < _k; j++ ) {
			theta.p(j) = theta.n(j) / n;
		}

		// update mean-subtracted data array
		theta.subtract_means(X);

		// compute S_j = sum((x_i - mu_j) * (x_i - mu_j)', x_i in cluster j) / n_j
		const auto& Xsubs = theta.Xsubs();

		for ( int j = 0; j < _k; j++ ) {
			Matrix& S_j = theta.S(j);
			S_j.init_zeros();

			for ( int i = 0; i < n; i++ ) {
				if ( y[i] == j ) {
					S_j.gemm(1 / theta.n(j), Xsubs[j][i], Xsubs[j][i].T(), 1.0f);
				}
			}
		}

		// update pdf matrix
		theta.pdf_all();

		// save outputs
		_entropy = 0;
		_log_likelihood = theta.log_likelihood();
		_num_parameters = _k * (1 + d + d * d);
		_num_samples = n;
		_output = y;
	}
	catch ( std::runtime_error& e ) {
		status = 1;
	}

	Timer::pop();

	return status;
}

/**
 * Print a k-means layer.
 */
void KMeansLayer::print() const
{
	log(LL_VERBOSE, "K-means");
	log(LL_VERBOSE, "  %-20s  %10d", "k", _k);
}

}
