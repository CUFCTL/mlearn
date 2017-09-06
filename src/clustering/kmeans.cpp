/**
 * @file clustering/kmeans.cpp
 *
 * Implementation of k-means clustering.
 */
#include <cstdlib>
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

	std::vector<int> y(X.cols());

	timer_push("initialize means");

	std::vector<Matrix> means;

	for ( int i = 0; i < this->_k; i++ ) {
		int j = lrand48() % X.cols();

		means.push_back(X(j, j + 1));
	}

	timer_pop();

	while ( true ) {
		timer_push("assignment step");

		std::vector<int> y_next(X.cols());

		for ( int i = 0; i < X.cols(); i++ ) {
			int min_index = -1;
			precision_t min_dist;

			for ( int j = 0; j < this->_k; j++ ) {
				precision_t dist = m_dist_L2(X, i, means[j], 0);

				if ( min_index == -1 || dist < min_dist ) {
					min_index = j;
					min_dist = dist;
				}
			}

			y_next[i] = min_index;
		}

		timer_pop();

		if ( y == y_next ) {
			break;
		}

		y = y_next;

		timer_push("update step");

		for ( int i = 0; i < this->_k; i++ ) {
			Matrix mean = Matrix::zeros("", X.rows(), 1);
			int num = 0;

			for ( int j = 0; j < X.cols(); j++ ) {
				if ( y[j] == i ) {
					mean += X(j, j + 1);
					num++;
				}
			}

			mean /= num;

			means[i] = mean;
		}

		timer_pop();
	}

	timer_push("compute log-likelihood");

	precision_t L = 0;

	for ( int i = 0; i < this->_k; i++ ) {
		// estimate variance for cluster i
		precision_t variance = 0;
		int R_n = 0;

		for ( int j = 0; j < X.cols(); j++ ) {
			if ( y[j] == i ) {
				variance += m_dist_L2(X, j, means[i], 0);
				R_n++;
			}
		}

		variance /= R_n - 1;

		// compute log-likelihood for cluster i
		int K = this->_k;
		int M = X.rows();
		int R = X.cols();
		precision_t L_i = -R_n / 2 * logf(2 * M_PI) - R_n * M / 2 * logf(variance) - (R_n - K) / 2 + R_n * logf(R_n) - R_n * logf(R);

		L += L_i;
	}

	timer_pop();

	// save outputs
	this->_log_likelihood = L;
	this->_num_parameters = this->_k * X.rows() + this->_k;
	this->_num_samples = X.cols();
	this->_output = y;

	timer_pop();
}

/**
 * Print a k-means layer.
 */
void KMeansLayer::print() const
{
	log(LL_INFO, "k-means");
	log(LL_INFO, "  k: %d", this->_k);
	log(LL_INFO, "");
}

}
