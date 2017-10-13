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
 * Compute the value of a (spherical) Gaussian distribution at x:
 *
 *   h(x | mu, var) = (2pi / var)^(d/2) e^(-(x - mu)' * (x - mu) / (2 * var))
 *
 * @param x
 * @param mu
 * @param var
 */
float pdf(Matrix x, const Matrix& mu, float var)
{
	x -= mu;

	return powf(2.0f * M_PI / var, x.rows() / 2.0f) * expf(-x.dot(x) / (2.0f * var));
}

/**
 * Partition a matrix X of observations into
 * clusters using k-means clustering.
 *
 * @param X
 */
int KMeansLayer::compute(const std::vector<Matrix>& X)
{
	Timer::push("k-means");

	int n = X.size();
	int d = X[0].rows();
	std::vector<int> y;

	Timer::push("initialize means");

	std::vector<Matrix> mu = m_random_sample(X, this->_k);

	Timer::pop();

	while ( true ) {
		Timer::push("assignment step");

		std::vector<int> y_next(n);

		for ( int i = 0; i < n; i++ ) {
			// compute y_i = argmin(j, ||x_i - mu_j||)
			int min_j = -1;
			float min_dist;

			for ( int j = 0; j < this->_k; j++ ) {
				float dist = m_dist_L2(X[i], 0, mu[j], 0);

				if ( min_j == -1 || dist < min_dist ) {
					min_j = j;
					min_dist = dist;
				}
			}

			y_next[i] = min_j;
		}

		Timer::pop();

		if ( y == y_next ) {
			break;
		}

		y = y_next;

		Timer::push("update step");

		for ( int j = 0; j < this->_k; j++ ) {
			// compute mu_j = mean of all x_i in cluster j
			mu[j] = Matrix::zeros(d, 1);
			int n_j = 0;

			for ( int i = 0; i < n; i++ ) {
				if ( y[i] == j ) {
					mu[j] += X[i];
					n_j++;
				}
			}

			mu[j] /= n_j;
		}

		Timer::pop();
	}

	Timer::push("compute log-likelihood");

	// compute L = sum(L_j, j=1:k)
	float L = 0;

	for ( int j = 0; j < this->_k; j++ ) {
		// compute variance for cluster j
		float var_j = 0;
		float n_j = 0;

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				var_j += m_dist_L2(X[i], 0, mu[j], 0);
				n_j++;
			}
		}

		var_j /= n_j;

		// compute mixture proportion for cluster j
		float p_j = n_j / n;

		// compute L_j = log(sum(p_j * h(x_i | mu_j, var_j), i=1:n))
		float sum = 0;
		for ( int i = 0; i < n; i++ ) {
			sum += p_j * pdf(X[i], mu[j], var_j);
		}

		L += logf(sum);
	}

	Timer::pop();

	// save outputs
	this->_log_likelihood = L;
	this->_num_parameters = this->_k * (2 + d);
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
	log(LL_VERBOSE, "k-means");
	log(LL_VERBOSE, "  %-20s  %10d", "k", this->_k);
}

}
