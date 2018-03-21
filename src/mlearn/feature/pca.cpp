/**
 * @file feature/pca.cpp
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "mlearn/feature/pca.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace ML {



/**
 * Construct a PCA layer.
 *
 * @param n1
 */
PCALayer::PCALayer(int n1)
{
	_n1 = n1;
}



/**
 * Compute the principal components of a matrix X, which
 * consists of observations in rows or columns. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param X
 * @param y
 * @param c
 */
void PCALayer::compute(const Matrix& X, const std::vector<int>& y, int c)
{
	// if n1 = -1, use default value
	int n1 = (_n1 == -1)
		? std::min(X.rows(), X.cols())
		: _n1;

	Timer::push("PCA");

	if ( X.rows() > X.cols() ) {
		Timer::push("compute surrogate of covariance matrix L");

		Matrix L = X.T() * X;

		Timer::pop();

		Timer::push("compute eigendecomposition of L");

		Matrix V;
		L.eigen(n1, V, _D);

		Timer::pop();

		Timer::push("compute principal components");

		_W = X * V;

		Timer::pop();
	}
	else {
		Timer::push("compute covariance matrix C");

		Matrix C = X * X.T();

		Timer::pop();

		Timer::push("compute eigendecomposition of C");

		C.eigen(n1, _W, _D);

		Timer::pop();
	}

	Timer::pop();
}



/**
 * Project a matrix X into the feature space of a PCA layer.
 *
 * @param X
 */
Matrix PCALayer::project(const Matrix& X)
{
	return _W.T() * X;
}



/**
 * Save a PCA layer to a file.
 *
 * @param file
 */
void PCALayer::save(IODevice& file) const
{
	file << _W;
	file << _D;
}



/**
 * Load a PCA layer from a file.
 *
 * @param file
 */
void PCALayer::load(IODevice& file)
{
	file >> _W;
	file >> _D;
}



/**
 * Print information about a PCA layer.
 */
void PCALayer::print() const
{
	log(LL_VERBOSE, "PCA");
	log(LL_VERBOSE, "  %-20s  %10d", "n1", _n1);
}



}
