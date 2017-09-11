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
precision_t pdf(Matrix x, const Matrix& mu, precision_t var)
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
void KMeansLayer::compute(const Matrix& X)
{
	timer_push("k-means");

	int n = X.cols();
	int d = X.rows();
	std::vector<Matrix> X_col = m_copy_columns(X);
	std::vector<int> y;

	timer_push("initialize means");

	std::vector<Matrix> mu = m_random_sample(X, this->_k);

	timer_pop();

	while ( true ) {
		timer_push("assignment step");

		std::vector<int> y_next(n);

		for ( int i = 0; i < n; i++ ) {
			// compute y_i = argmin(j, ||x_i - mu_j||)
			int min_j = -1;
			precision_t min_dist;

			for ( int j = 0; j < this->_k; j++ ) {
				precision_t dist = m_dist_L2(X_col[i], 0, mu[j], 0);

				if ( min_j == -1 || dist < min_dist ) {
					min_j = j;
					min_dist = dist;
				}
			}

			y_next[i] = min_j;
		}

		timer_pop();

		if ( y == y_next ) {
			break;
		}

		y = y_next;

		timer_push("update step");

		for ( int j = 0; j < this->_k; j++ ) {
			// compute mu_j = mean of all x_i in cluster j
			mu[j] = Matrix::zeros(d, 1);
			int n_j = 0;

			for ( int i = 0; i < n; i++ ) {
				if ( y[i] == j ) {
					mu[j] += X_col[i];
					n_j++;
				}
			}

			mu[j] /= n_j;
		}

		timer_pop();
	}

	timer_push("compute log-likelihood");

	// compute L = sum(L_j, j=1:k)
	precision_t L = 0;

	for ( int j = 0; j < this->_k; j++ ) {
		// compute variance for cluster j
		precision_t var_j = 0;
		precision_t n_j = 0;

		for ( int i = 0; i < n; i++ ) {
			if ( y[i] == j ) {
				var_j += m_dist_L2(X_col[i], 0, mu[j], 0);
				n_j++;
			}
		}

		var_j /= n_j;

		// compute mixture proportion for cluster j
		precision_t p_j = n_j / n;

		// compute L_j = log(sum(p_j * h(x_i | mu_j, var_j), i=1:n))
		precision_t sum = 0;
		for ( int i = 0; i < n; i++ ) {
			sum += p_j * pdf(X_col[i], mu[j], var_j);
		}

		L += logf(sum);
	}

	timer_pop();

	// save outputs
	this->_log_likelihood = L;
	this->_num_parameters = this->_k * (2 + d);
	this->_num_samples = n;
	this->_output = y;

	timer_pop();
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
