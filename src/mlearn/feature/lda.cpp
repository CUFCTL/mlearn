/**
 * @file feature/lda.cpp
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include "mlearn/feature/lda.h"
#include "mlearn/feature/pca.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace mlearn {



/**
 * Construct an LDA layer.
 *
 * @param n1
 * @param n2
 */
LDALayer::LDALayer(int n1, int n2)
{
	_n1 = n1;
	_n2 = n2;
}



/**
 * Compute the LDA features of a matrix X.
 *
 * @param X
 * @param y
 * @param c
 */
void LDALayer::fit(const Matrix& X, const std::vector<int>& y, int c)
{
	// if n1 = -1, use default value
	int n1 = (_n1 == -1)
		? X.cols() - c
		: _n1;

	// if n2 = -1, use default value
	int n2 = (_n2 == -1)
		? c - 1
		: _n2;

	if ( n1 <= 0 ) {
		Logger::log(LogLevel::Error, "error: training set is too small for LDA");
		exit(1);
	}

	Timer::push("LDA");

	Timer::push("compute eigenfaces");

	PCALayer pca(n1);
	pca.fit(X, y, c);
	Matrix P_pca = pca.transform(X);

	Timer::pop();

	Timer::push("compute scatter matrices S_b and S_w");

	std::vector<Matrix> X_c = m_copy_classes(P_pca, y, c);
	std::vector<Matrix> U = m_class_means(X_c);
	Matrix S_b = m_scatter_between(X_c, U);
	Matrix S_w = m_scatter_within(X_c, U);

	Timer::pop();

	Timer::push("compute eigendecomposition of S_b and S_w");

	Matrix S_w_inv = S_w.inverse();
	Matrix J = S_w_inv * S_b;

	Matrix W_fld;
	Matrix J_eval;
	J.eigen(n2, W_fld, J_eval);

	Timer::pop();

	Timer::push("compute Fisherfaces");

	_W = pca.W() * W_fld;

	Timer::pop();

	Timer::pop();
}



/**
 * Project a matrix X into the feature space of an LDA layer.
 *
 * @param X
 */
Matrix LDALayer::transform(const Matrix& X)
{
	return _W.T() * X;
}



/**
 * Save an LDA layer to a file.
 *
 * @param file
 */
void LDALayer::save(IODevice& file) const
{
	file << _n1;
	file << _n2;
	file << _W;
}



/**
 * Load an LDA layer from a file.
 *
 * @param file
 */
void LDALayer::load(IODevice& file)
{
	file >> _n1;
	file >> _n2;
	file >> _W;
}



/**
 * Print information about an LDA layer.
 */
void LDALayer::print() const
{
	Logger::log(LogLevel::Verbose, "LDA");
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "n1", _n1);
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "n2", _n2);
}



}
