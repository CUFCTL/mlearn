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

		for ( int i = 0; i < this->_k; i++ ) {
			mu[i] = Matrix::zeros(d, 1);
			int num = 0;

			for ( int j = 0; j < n; j++ ) {
				if ( y[j] == i ) {
					mu[i] += X_col[j];
					num++;
				}
			}

			mu[i] /= num;
		}

		timer_pop();
	}

	timer_push("compute log-likelihood");

	precision_t L = 0;

	for ( int i = 0; i < this->_k; i++ ) {
		// estimate variance for cluster i
		precision_t variance = 0;
		int R_n = 0;

		for ( int j = 0; j < n; j++ ) {
			if ( y[j] == i ) {
				variance += m_dist_L2(X_col[j], 0, mu[i], 0);
				R_n++;
			}
		}

		variance /= R_n - 1;

		// compute log-likelihood for cluster i
		int K = this->_k;
		int M = d;
		int R = n;
		precision_t L_i = -R_n / 2 * logf(2 * M_PI) - R_n * M / 2 * logf(variance) - (R_n - K) / 2 + R_n * logf(R_n) - R_n * logf(R);

		L += L_i;
	}

	timer_pop();

	// save outputs
	this->_log_likelihood = L;
	this->_num_parameters = this->_k * d + this->_k;
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
