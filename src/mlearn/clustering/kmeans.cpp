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
 * @param K
 */
KMeansLayer::KMeansLayer(int K)
{
	_K = K;
}



/**
 * Fit a clustering model to a dataset X using k-means.
 *
 * @param X
 */
void KMeansLayer::fit(const std::vector<Matrix>& X)
{
	Timer::push("K-means");

	int N = X.size();
	int D = X[0].rows();

	// initialize means randomly from X
	std::vector<Matrix> means = m_random_sample(X, _K);

	// iterate k means until convergence
	std::vector<int> y(N);
	std::vector<int> y_next(N);

	while ( true ) {
		// compute new labels
		for ( int i = 0; i < N; i++ ) {
			// find k that minimizes norm(x_i - mu_k)
			int min_k = -1;
			float min_dist;

			for ( int k = 0; k < _K; k++ ) {
				float dist = m_dist_L2(X[i], 0, means[k], 0);

				if ( min_k == -1 || dist < min_dist ) {
					min_k = k;
					min_dist = dist;
				}
			}

			y_next[i] = min_k;
		}

		// check for convergence
		if ( y == y_next ) {
			break;
		}

		// update labels
		std::swap(y, y_next);

		// update means
		for ( int k = 0; k < _K; k++ ) {
			// compute mu_k = mean of all x_i in cluster k
			int n_k = 0;

			means[k].init_zeros();

			for ( int i = 0; i < N; i++ ) {
				if ( y[i] == k ) {
					means[k] += X[i];
					n_k++;
				}
			}
			means[k] /= n_k;
		}
	}

	// compute within-class scatter
	float S = 0;

	for ( int k = 0; k < _K; k++ ) {
		for ( int i = 0; i < N; i++ ) {
			if ( y[i] == k ) {
				float dist = m_dist_L2(X[i], 0, means[k], 0);

				S += dist * dist;
			}
		}
	}

	// save outputs
	_log_likelihood = -S;
	_num_parameters = _K * D;
	_num_samples = N;
	_labels = y;

	Timer::pop();
}



/**
 * Print a k-means layer.
 */
void KMeansLayer::print() const
{
	Logger::log(LogLevel::Verbose, "K-means");
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "K", _K);
}



}
