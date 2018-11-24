/**
 * @file feature/ica.cpp
 *
 * Implementation of ICA (Hyvarinen, 1999).
 */
#include <cmath>
#include "mlearn/feature/ica.h"
#include "mlearn/feature/pca.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace mlearn {



typedef Matrix (*fpica_func_t)(const Matrix&, const Matrix&);



/**
 * Construct an ICA layer.
 *
 * @param n1
 * @param n2
 * @param nonl
 * @param max_iter
 * @param eps
 */
ICALayer::ICALayer(int n1, int n2, ICANonl nonl, int max_iter, float eps)
{
	_n1 = n1;
	_n2 = n2;
	_nonl = nonl;
	_max_iter = max_iter;
	_eps = eps;
}



/**
 * Compute the independent components of a matrix X, which
 * consists of observations in columns.
 *
 * @param X
 * @param y
 * @param c
 */
void ICALayer::fit(const Matrix& X, const std::vector<int>& y, int c)
{
	Timer::push("ICA");

	Timer::push("subtract mean from input matrix");

	// compute mixedsig = X', subtract mean column
	Matrix mixedsig = X.transpose();
	Matrix mixedmean = mixedsig.mean_column();

	mixedsig.subtract_columns(mixedmean);

	Timer::pop();

	Timer::push("compute whitening matrix and whitened input matrix");

	// compute whitening matrix W_z = inv(sqrt(D)) * W_pca'
	PCALayer pca(_n1);

	pca.fit(mixedsig, y, c);

	Matrix D = pca.D();
	D.elem_apply(sqrtf);
	D = D.inverse();

	Matrix W_z = D * pca.W().T();

	// compute whitened input U = W_z * mixedsig
	Matrix U = W_z * mixedsig;

	Timer::pop();

	Timer::push("compute mixing matrix");

	// compute mixing matrix
	Matrix W_mix = fpica(U, W_z);

	Timer::pop();

	Timer::push("compute ICA projection matrix");

	// compute independent components
	// icasig = W_mix * (mixedsig + mixedmean * ones(1, mixedsig.cols()))
	Matrix icasig_temp1 = mixedmean * Matrix::ones(1, mixedsig.cols());
	icasig_temp1 += mixedsig;

	Matrix icasig = W_mix * icasig_temp1;

	// compute W_ica = icasig'
	_W = icasig.transpose();

	Timer::pop();

	Timer::pop();
}



/**
 * Project a matrix X into the feature space of an ICA layer.
 *
 * @param X
 */
Matrix ICALayer::transform(const Matrix& X)
{
	return _W.T() * X;
}



/**
 * Save an ICA layer to a file.
 *
 * @param file
 */
void ICALayer::save(IODevice& file) const
{
	file << _n1;
	file << _n2;
	file << (int) _nonl;
	file << _max_iter;
	file << _eps;
	file << _W;
}



/**
 * Load an ICA layer from a file.
 *
 * @param file
 */
void ICALayer::load(IODevice& file)
{
	file >> _n1;
	file >> _n2;
	int nonl; file >> nonl; _nonl = (ICANonl) nonl;
	file >> _max_iter;
	file >> _eps;
	file >> _W;
}



/**
 * Print information about an ICA layer.
 */
void ICALayer::print() const
{
	const char *nonl_name = "";

	if ( _nonl == ICANonl::pow3 ) {
		nonl_name = "pow3";
	}
	else if ( _nonl == ICANonl::tanh ) {
		nonl_name = "tanh";
	}
	else if ( _nonl == ICANonl::gauss ) {
		nonl_name = "gauss";
	}

	Logger::log(LogLevel::Verbose, "ICA");
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "n1", _n1);
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "n2", _n2);
	Logger::log(LogLevel::Verbose, "  %-20s  %10s", "nonl", nonl_name);
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "max_iter", _max_iter);
	Logger::log(LogLevel::Verbose, "  %-20s  %10f", "eps", _eps);
}



/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 */
float pow3(float x)
{
	return pow(x, 3);
}



/**
 * Compute the parameter update for fpica
 * with the pow3 nonlinearity:
 *
 *   g(u) = u^3
 *   g'(u) = 3 * u^2
 *
 * which gives:
 *
 *   w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X.cols()
 *      = X * ((X' * w) .^ 3) / X.cols() - 3 * w
 *
 * @param w0
 * @param X
 */
Matrix fpica_pow3 (const Matrix& w0, const Matrix& X)
{
	Matrix w_temp1 = X.T() * w0;
	w_temp1.elem_apply(pow3);

	Matrix w = X * w_temp1;
	w /= X.cols();
	w -= 3 * w0;

	return w;
}



/**
 * Derivative of tanh nonlinearity function.
 *
 * @param x
 */
float tanh_deriv(float x)
{
	return pow(1 / cosh(x), 2);
}



/**
 * Compute the parameter update for fpica
 * with the tanh nonlinearity:
 *
 *   g(u) = tanh(u)
 *   g'(u) = sech(u)^2 = 1 - tanh(u)^2
 *
 * which gives:
 *
 *   w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X.cols()
 *
 * @param w0
 * @param X
 */
Matrix fpica_tanh(const Matrix& w0, const Matrix& X)
{
	Matrix w_temp1 = X.T() * w0;
	Matrix w_temp2 = w_temp1;

	w_temp1.elem_apply(tanhf);
	w_temp2.elem_apply(tanh_deriv);

	Matrix w = X * w_temp1;
	w -= w_temp2.sum() * w0;
	w /= X.cols();

	return w;
}



/**
 * Gaussian nonlinearity function.
 *
 * @param x
 */
float gauss(float x)
{
	return x * exp(-(x * x) / 2.0f);
}



/**
 * Derivative of the Gaussian nonlinearity function.
 *
 * @param x
 */
float gauss_deriv(float x)
{
	return (1 - x * x) * exp(-(x * x) / 2.0f);
}



/**
 * Compute the parameter update for fpica
 * with the Gaussian nonlinearity:
 *
 *   g(u) = u * exp(-u^2 / 2)
 *   g'(u) = (1 - u^2) * exp(-u^2 / 2)
 *
 * which gives:
 *
 *   w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X.cols()
 *
 * @param w0
 * @param X
 */
Matrix fpica_gauss(const Matrix& w0, const Matrix& X)
{
	Matrix w_temp1 = X.T() * w0;
	Matrix w_temp2 = w_temp1;

	w_temp1.elem_apply(gauss);
	w_temp2.elem_apply(gauss_deriv);

	Matrix w = X * w_temp1;
	w -= w_temp2.sum() * w0;
	w /= X.cols();

	return w;
}



/**
 * Compute the mixing matrix W_mix for an input matrix X using
 * the deflation approach. The input matrix should already
 * be whitened.
 *
 * @param X
 * @param W_z
 */
Matrix ICALayer::fpica(const Matrix& X, const Matrix& W_z)
{
	// if n2 is -1, use default value
	int n2 = (_n2 == -1)
		? X.rows()
		: std::min(X.rows(), _n2);

	// determine nonlinearity function
	fpica_func_t fpica_update = nullptr;

	if ( _nonl == ICANonl::pow3 ) {
		fpica_update = fpica_pow3;
	}
	else if ( _nonl == ICANonl::tanh ) {
		fpica_update = fpica_tanh;
	}
	else if ( _nonl == ICANonl::gauss ) {
		fpica_update = fpica_gauss;
	}

	Matrix B = Matrix::zeros(n2, n2);
	Matrix W_mix = Matrix::zeros(n2, W_z.cols());

	int i;
	for ( i = 0; i < n2; i++ ) {
		Logger::log(LogLevel::Verbose, "      round %d", i + 1);

		// initialize w as a Gaussian (0, 1) random vector
		Matrix w = Matrix::random(n2, 1);

		// compute w = w - B * B' * w, normalize w
		w -= B * B.T() * w;
		w /= w.nrm2();

		// initialize w0
		Matrix w0 = Matrix::zeros(w.rows(), w.cols());

		int j;
		for ( j = 0; j < _max_iter; j++ ) {
			// compute w = w - B * B' * w, normalize w
			w -= B * B.T() * w;
			w /= w.nrm2();

			// determine whether the direction of w and w0 are equal
			float norm1 = (w - w0).nrm2();
			float norm2 = (w + w0).nrm2();

			// terminate round if w converges
			if ( norm1 < _eps || norm2 < _eps ) {
				// save B(:, i) = w
				B.assign_column(i, w, 0);

				// save W_mix(i, :) = w' * W_z
				W_mix.assign_row(i, w.T() * W_z, 0);

				// continue to the next round
				break;
			}

			// update w0
			w0.assign_column(0, w, 0);

			// compute w+ based on non-linearity
			w = fpica_update(w0, X);

			// compute w* = w+ / ||w+||
			w /= w.nrm2();
		}

		Logger::log(LogLevel::Verbose, "      iterations: %d", j);
	}

	return W_mix;
}



}
