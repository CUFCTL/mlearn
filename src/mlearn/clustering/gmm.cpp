/**
 * @file clustering/gmm.cpp
 *
 * Implementation of Gaussian mixture models.
 */
#include <cmath>
#include <stdexcept>
#include "mlearn/clustering/gmm.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"

namespace ML {



/**
 * Construct a GMM layer.
 *
 * @param K
 */
GMMLayer::GMMLayer(int K)
{
	_K = K;
	_success = false;
}



void GMMLayer::Component::initialize(float pi, const Matrix& mu)
{
	const int D = mu.rows();

	// initialize pi and mu as given
	this->pi = pi;
	this->mu = mu;

	// use identity covariance- assume dimensions are independent
	this->sigma = Matrix::identity(D);

	// initialize zero artifacts
	_sigma_inv = Matrix::zeros(D, D);
	_normalizer = 0;
}



void GMMLayer::Component::prepare()
{
	const int D = mu.rows();

	// compute inverse of sigma
	_sigma_inv = sigma.inverse();

	// compute normalizer for multivariate normal distribution
	float det = sigma.determinant();

	_normalizer = -0.5f * (D * log(2.0f * M_PI) + log(det));
}



/**
 * Compute the probability density function of the multivariate
 * normal distribution conditioned on a single component for each
 * data point in X:
 *
 *   P(x|k) = exp(-0.5 * (x - mu)^T S^-1 (x - mu)) / sqrt((2pi)^D det(S))
 *
 * @param X
 * @param logP
 * @param k
 */
void GMMLayer::Component::compute_log_mv_norm(const std::vector<Matrix>& X, Matrix& logP, int k)
{
	const int N = X.size();

	for ( int i = 0; i < N; i++ )
	{
		// compute xm = (x_i - mu)
		Matrix xm = X[i];
		xm -= mu;

		// compute log(P(x_i|k)) = normalizer - 0.5 * xm^T * S^-1 * xm
		logP.elem(k, i) = _normalizer - 0.5f * xm.dot(_sigma_inv * xm);
	}
}



void GMMLayer::kmeans(const std::vector<Matrix>& X)
{
	const int N = X.size();
	const int MAX_ITERATIONS = 20;
	const float TOLERANCE = 1e-3;
	float diff = 0;

	std::vector<Matrix> MP(_K);
	std::vector<int> counts(_K);

	for ( int t = 0; t < MAX_ITERATIONS && diff > TOLERANCE; t++ )
	{
		for ( int k = 0; k < _K; k++ )
		{
			MP[k].init_zeros();
		}

		counts.assign(counts.size(), 0);

		for ( int i = 0; i < N; i++ )
		{
			float min_dist = INFINITY;
			int min_k = 0;

			for ( int k = 0; k < _K; k++ )
			{
				float dist = m_dist_L2(X[i], 0, _components[k].mu, 0);
				if ( min_dist > dist )
				{
					min_dist = dist;
					min_k = k;
				}
			}

			MP[min_k] += X[i];
			counts[min_k]++;
		}

		for ( int k = 0; k < _K; k++ )
		{
			MP[k] /= counts[k];
		}

		diff = 0;
		for ( int k = 0; k < _K; k++ )
		{
			diff += m_dist_L2(MP[k], 0, _components[k].mu, 0);
		}
		diff /= _K;

		for ( int k = 0; k < _K; k++ )
		{
			_components[k].mu = MP[k];
		}
	}
}



void GMMLayer::compute_log_pi(Matrix& logpi)
{
	for ( int k = 0; k < _K; k++ )
	{
		logpi.elem(k) = log(_components[k].pi);
	}
}



void GMMLayer::compute_log_mv_norm(const std::vector<Matrix>& X, Matrix& loggamma)
{
	const int N = X.size();

	for ( int k = 0; k < _K; k++ )
	{
		_components[k].compute_log_mv_norm(X, loggamma, k);
	}
}



void GMMLayer::compute_log_gamma_nk(const Matrix& logpi, Matrix& loggamma, float& logL)
{
	const int N = loggamma.cols();

	logL = 0;
	for ( int i = 0; i < N; i++ )
	{
		float maxArg = -INFINITY;
		for ( int k = 0; k < _K; k++ )
		{
			float logProbK = logpi.elem(k) + loggamma.elem(k, i);
			if ( logProbK > maxArg )
			{
				maxArg = logProbK;
			}
		}

		float sum = 0;
		for ( int k = 0; k < _K; k++ )
		{
			float logProbK = logpi.elem(k) + loggamma.elem(k, i);
			sum += exp(logProbK - maxArg);
		}

		float logpx = maxArg + log(sum);
		for ( int k = 0; k < _K; k++ )
		{
			loggamma.elem(k, i) += -logpx;
		}

		logL += logpx;
	}
}



void GMMLayer::compute_log_gamma_k(const Matrix& loggamma, Matrix& logGamma)
{
	const int N = loggamma.cols();

	for ( int k = 0; k < _K; k++ )
	{
		float maxArg = -INFINITY;
		for ( int i = 0; i < N; i++ )
		{
			float loggammank = loggamma.elem(k, i);
			if ( loggammank > maxArg )
			{
				maxArg = loggammank;
			}
		}

		float sum = 0;
		for ( int i = 0; i < N; i++ )
		{
			float loggammank = loggamma.elem(k, i);
			sum += exp(loggammank - maxArg);
		}

		logGamma.elem(k) = maxArg + log(sum);
	}
}



float GMMLayer::compute_log_gamma_sum(const Matrix& logpi, const Matrix& logGamma)
{
	float maxArg = -INFINITY;
	for ( int k = 0; k < _K; k++ )
	{
		float arg = logpi.elem(k) + logGamma.elem(k);
		if ( arg > maxArg )
		{
			maxArg = arg;
		}
	}

	float sum = 0;
	for ( int k = 0; k < _K; k++ )
	{
		float arg = logpi.elem(k) + logGamma.elem(k);
		sum += exp(arg - maxArg);
	}

	return maxArg + log(sum);
}



void GMMLayer::update(Matrix& logpi, Matrix& loggamma, Matrix& logGamma, float logGammaSum, const std::vector<Matrix>& X)
{
	const int N = X.size();

	// update logpi
	for ( int k = 0; k < _K; k++ )
	{
		logpi.elem(k) += logGamma.elem(k) - logGammaSum;
	}

	// convert loggamma / logGamma to gamma / Gamma to avoid duplicate exp(x) calls
	loggamma.elem_apply(exp);
	logGamma.elem_apply(exp);

	// update model parameters
	const Matrix& gamma {loggamma};
	const Matrix& Gamma {logGamma};

	for ( int k = 0; k < _K; k++ )
	{
		// update pi
		_components[k].pi = exp(logpi.elem(k));

		// update mu
		Matrix& mu = _components[k].mu;
		mu.init_zeros();

		for ( int i = 0; i < N; i++ )
		{
			mu.axpy(gamma.elem(k, i), X[i]);
		}

		mu /= Gamma.elem(k);

		// update sigma
		Matrix& sigma = _components[k].sigma;
		sigma.init_zeros();

		for ( int i = 0; i < N; i++ )
		{
			// compute xm = (x - mu)
			Matrix xm = X[i];
			xm -= mu;

			// compute S_i = gamma_ik * (x - mu) (x - mu)^T
			sigma.gemm(gamma.elem(k, i), xm, xm.T(), 1.0f);
		}

		sigma /= Gamma.elem(k);

		_components[k].prepare();
	}
}



std::vector<int> GMMLayer::compute_labels(const Matrix& gamma)
{
	const int N = gamma.cols();
	std::vector<int> labels(N);

	for ( int i = 0; i < N; i++ )
	{
		int max_k = -1;
		float max_gamma = -INFINITY;

		for ( int k = 0; k < _K; k++ )
		{
			if ( max_gamma < gamma.elem(k, i) )
			{
				max_k = k;
				max_gamma = gamma.elem(k, i);
			}
		}

		labels[i] = max_k;
	}

	return labels;
}



float GMMLayer::compute_entropy(const Matrix& gamma, const std::vector<int>& labels)
{
	const int N = gamma.cols();
	float E = 0;

	for ( int i = 0; i < N; i++ )
	{
		E += log(gamma.elem(labels[i], i));
	}

	return E;
}



/**
 * Fit a Gaussian mixture model to a dataset.
 *
 * @param X
 */
void GMMLayer::fit(const std::vector<Matrix>& X)
{
	Timer::push("Gaussian mixture model");

	int N = X.size();
	int D = X[0].rows();

	// initialize components
	_components.resize(_K);

	for ( int k = 0; k < _K; k++ )
	{
		// use uniform mixture proportion and randomly sampled mean
		int i = rand() % N;

		_components[k].initialize(1.0f / _K, X[i]);
		_components[k].prepare();
	}

	// initialize means with k-means
	kmeans(X);

	// initialize workspace
	Matrix logpi(_K, 1);
	Matrix loggamma(_K, N);
	Matrix logGamma(_K, 1);

	compute_log_pi(logpi);

	// run EM algorithm
	const int MAX_ITERATIONS = 100;
	const float TOLERANCE = 1e-8;
	float L_prev = -INFINITY;
	float L = -INFINITY;

	try
	{
		for ( int t = 0; t < MAX_ITERATIONS; t++ )
		{
			// E step
			// compute gamma, log-likelihood
			compute_log_mv_norm(X, loggamma);

			L_prev = L;
			compute_log_gamma_nk(logpi, loggamma, L);

			// check for convergence
			if ( fabs(L - L_prev) < TOLERANCE )
			{
				Logger::log(LogLevel::Debug, "converged after %d iterations", t + 1);
				break;
			}

			// M step
			// compute Gamma_k = sum(gamma_ik, i=1:N)
			compute_log_gamma_k(loggamma, logGamma);

			float logGammaSum = compute_log_gamma_sum(logpi, logGamma);

			// update parameters
			update(logpi, loggamma, logGamma, logGammaSum, X);
		}

		// save outputs
		_log_likelihood = L;
		_num_parameters = _K * (1 + D + D * D);
		_num_samples = N;
		_entropy = compute_entropy(loggamma, compute_labels(loggamma));
		_success = true;
	}
	catch ( std::runtime_error& e ) {
		_success = false;
	}

	Timer::pop();
}



/**
 * Predict a set of labels for a dataset.
 *
 * @param X
 */
std::vector<int> GMMLayer::predict(const std::vector<Matrix>& X)
{
	const int N = X.size();
	Matrix logpi(_K, 1);
	Matrix loggamma(_K, N);
	float L;

	compute_log_pi(logpi);
	compute_log_mv_norm(X, loggamma);
	compute_log_gamma_nk(logpi, loggamma, L);

	return compute_labels(loggamma);
}



/**
 * Print a GMM layer.
 */
void GMMLayer::print() const
{
	Logger::log(LogLevel::Verbose, "Gaussian mixture model");
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "K", _K);
}



}
