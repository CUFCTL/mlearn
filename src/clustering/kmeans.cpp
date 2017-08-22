/**
 * @file clustering/kmeans.cpp
 *
 * Implementation of k-means clustering.
 */
#include <cstdlib>
#include "clustering/kmeans.h"
#include "math/matrix_utils.h"

namespace ML {

/**
 * Partition a data matrix X of observations into
 * k clusters using k-means clustering.
 *
 * @param X
 * @param k
 */
std::vector<int> KMeansLayer::compute(const Matrix& X, int k)
{
	std::vector<int> y(X.cols());

	// initialization step (Forgy)
	std::vector<Matrix> means;

	for ( int i = 0; i < k; i++ ) {
		int j = lrand48() % X.cols();

		means.push_back(X(j, j + 1));
	}

	while ( true ) {
		// assignment step
		std::vector<int> y_next(X.cols());

		for ( int i = 0; i < X.cols(); i++ ) {
			int min_index = -1;
			precision_t min_dist;

			for ( int j = 0; j < k; j++ ) {
				precision_t dist = m_dist_L2(X, i, means[j], 0);

				if ( min_index == -1 || dist < min_dist ) {
					min_index = j;
					min_dist = dist;
				}
			}

			y_next[i] = min_index;
		}

		// check for convergence
		if ( y == y_next ) {
			break;
		}

		y = y_next;

		// update step
		for ( int i = 0; i < k; i++ ) {
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
	}

	return y;
}

}
