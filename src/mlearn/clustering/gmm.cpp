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

namespace mlearn {



IODevice& operator<<(IODevice& file, const GMMLayer::Component& component)
{
	file << component.pi;
	file << component.mu;
	file << component.sigma;
	file << component._sigma_inv;
	file << component._normalizer;
	return file;
}



IODevice& operator>>(IODevice& file, GMMLayer::Component& component)
{
	file >> component.pi;
	file >> component.mu;
	file >> component.sigma;
	file >> component._sigma_inv;
	file >> component._normalizer;
	return file;
}



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
void GMMLayer::Component::compute_log_prob(const std::vector<Matrix>& X, Matrix& logP, int k)
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



float GMMLayer::e_step(const std::vector<Matrix>& X, Matrix& gamma)
{
	const int N = X.size();

	// compute logpi
	Matrix logpi(_K, 1);

	for ( int k = 0; k < _K; k++ )
	{
		logpi.elem(k) = log(_components[k].pi);
	}

	// compute log-probability for each point in X and each cluster
	Matrix& logP = gamma;

	for ( int k = 0; k < _K; k++ )
	{
		_components[k].compute_log_prob(X, logP, k);
	}

	// compute loggamma and log-likelihood
	float logL = 0;

	for ( int i = 0; i < N; i++ )
	{
		float maxArg = -INFINITY;
		for ( int k = 0; k < _K; k++ )
		{
			float logProbK = logpi.elem(k) + logP.elem(k, i);
			if ( logProbK > maxArg )
			{
				maxArg = logProbK;
			}
		}

		float sum = 0;
		for ( int k = 0; k < _K; k++ )
		{
			float logProbK = logpi.elem(k) + logP.elem(k, i);
			sum += exp(logProbK - maxArg);
		}

		float logpx = maxArg + log(sum);
		for ( int k = 0; k < _K; k++ )
		{
			gamma.elem(k, i) += logpi.elem(k) - logpx;
		}

		logL += logpx;
	}

	// compute gamma
	gamma.elem_apply(expf);

	return logL;
}



void GMMLayer::m_step(const std::vector<Matrix>& X, const Matrix& gamma)
{
	const int N = X.size();

	for ( int k = 0; k < _K; k++ )
	{
		// compute n_k = sum(gamma_ki)
		float n_k = 0;

		for ( int i = 0; i < N; i++ )
		{
			n_k += gamma.elem(k, i);
		}

		// update pi
		_components[k].pi = n_k / N;

		// update mu
		Matrix& mu = _components[k].mu;
		mu.init_zeros();

		for ( int i = 0; i < N; i++ )
		{
			mu.axpy(gamma.elem(k, i), X[i]);
		}

		mu /= n_k;

		// update sigma
		Matrix& sigma = _components[k].sigma;
		sigma.init_zeros();

		for ( int i = 0; i < N; i++ )
		{
			// compute xm = (x - mu)
			Matrix xm = X[i];
			xm -= mu;

			// compute S_i = gamma_ki * (x - mu) (x - mu)^T
			sigma.gemm(gamma.elem(k, i), xm, xm.T(), 1.0f);
		}

		sigma /= n_k;

		// pre-compute precision matrix and normalizer
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
	Matrix gamma(_K, N);

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
			L_prev = L;
			L = e_step(X, gamma);

			// check for convergence
			if ( fabs(L - L_prev) < TOLERANCE )
			{
				Logger::log(LogLevel::Debug, "converged after %d iterations", t + 1);
				break;
			}

			// M step
			m_step(X, gamma);
		}

		// save outputs
		_log_likelihood = L;
		_num_parameters = _K * (1 + D + D * D);
		_num_samples = N;
		_entropy = compute_entropy(gamma, compute_labels(gamma));
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

	Matrix gamma(_K, N);
	e_step(X, gamma);

	return compute_labels(gamma);
}



/**
 * Save a GMM layer to a file.
 *
 * @param file
 */
void GMMLayer::save(IODevice& file) const
{
	file << _K;
	file << _components;
	file << _entropy;
	file << _log_likelihood;
	file << _num_parameters;
	file << _num_samples;
	file << _success;
}



/**
 * Load a GMM layer from a file.
 *
 * @param file
 */
void GMMLayer::load(IODevice& file)
{
	file >> _K;
	file >> _components;
	file >> _entropy;
	file >> _log_likelihood;
	file >> _num_parameters;
	file >> _num_samples;
	file >> _success;
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
