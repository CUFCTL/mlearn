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
   const int D = sigma.rows();

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
 */
void GMMLayer::Component::compute_log_mv_norm(const std::vector<Matrix>& X, float *logP)
{
	const int N = X.size();

   for ( int i = 0; i < N; i++ )
   {
      // compute xm = (x - mu)
      Matrix xm = X[i];
		xm -= mu;

      // compute log(P(x|k)) = normalizer - 0.5 * xm^T * S^-1 * xm
      logP[i] = _normalizer - 0.5f * xm.dot(_sigma_inv * xm);
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



void GMMLayer::compute_log_mv_norm(const std::vector<Matrix>& X, float *loggamma)
{
	const int N = X.size();

   for ( int k = 0; k < _K; k++ )
   {
      _components[k].compute_log_mv_norm(X, &loggamma[k * N]);
   }
}



void GMMLayer::compute_log_likelihood_gamma_nk(const float *logpi, int K, float *loggamma, int N, float *logL)
{
   *logL = 0.0;
   for ( int i = 0; i < N; i++ )
   {
      float maxArg = -INFINITY;
      for ( int k = 0; k < K; k++ )
      {
         const float logProbK = logpi[k] + loggamma[k * N + i];
         if ( logProbK > maxArg )
         {
            maxArg = logProbK;
         }
      }

      float sum = 0.0;
      for ( int k = 0; k < K; k++ )
      {
         const float logProbK = logpi[k] + loggamma[k * N + i];
         sum += exp(logProbK - maxArg);
      }

      const float logpx = maxArg + log(sum);
      *logL += logpx;
      for ( int k = 0; k < K; k++ )
      {
         loggamma[k * N + i] += -logpx;
      }
   }
}



void GMMLayer::compute_log_gamma_k(const float *loggamma, int N, int K, float *logGamma)
{
   memset(logGamma, 0, K * sizeof(float));

   for ( int k = 0; k < K; k++ )
   {
      const float *loggammak = &loggamma[k * N];

      float maxArg = -INFINITY;
      for ( int i = 0; i < N; i++ )
      {
         const float loggammank = loggammak[i];
         if ( loggammank > maxArg )
         {
            maxArg = loggammank;
         }
      }

      float sum = 0;
      for ( int i = 0; i < N; i++ )
      {
         const float loggammank = loggammak[i];
         sum += exp(loggammank - maxArg);
      }

      logGamma[k] = maxArg + log(sum);
   }
}



float GMMLayer::compute_log_gamma_sum(const float *logpi, int K, const float *logGamma)
{
   float maxArg = -INFINITY;
   for ( int k = 0; k < K; k++ )
   {
      const float arg = logpi[k] + logGamma[k];
      if ( arg > maxArg )
      {
         maxArg = arg;
      }
   }

   float sum = 0;
   for ( int k = 0; k < K; k++ )
   {
      const float arg = logpi[k] + logGamma[k];
      sum += exp(arg - maxArg);
   }

   return maxArg + log(sum);
}



void GMMLayer::update(float *logpi, int K, float *loggamma, float *logGamma, float logGammaSum, const std::vector<Matrix>& X)
{
	const int N = X.size();

   // update pi
   for ( int k = 0; k < K; k++ )
   {
      logpi[k] += logGamma[k] - logGammaSum;

      _components[k].pi = exp(logpi[k]);
   }

   // convert loggamma / logGamma to gamma / Gamma to avoid duplicate exp(x) calls
   for ( int k = 0; k < K; k++ )
   {
      for ( int i = 0; i < N; i++ )
      {
         const int idx = k * N + i;
         loggamma[idx] = exp(loggamma[idx]);
      }
   }

   for ( int k = 0; k < K; k++ )
   {
      logGamma[k] = exp(logGamma[k]);
   }

   for ( int k = 0; k < K; k++ )
   {
      // update mu
		Matrix& mu = _components[k].mu;
		mu.init_zeros();

      for ( int i = 0; i < N; i++ )
      {
			mu.axpy(loggamma[k * N + i], X[i]);
      }

		mu /= logGamma[k];

      // update sigma
      Matrix& sigma = _components[k].sigma;
		sigma.init_zeros();

      for ( int i = 0; i < N; i++ )
      {
         // compute xm = (x - mu)
         Matrix xm = X[i];
			xm -= mu;

         // compute S_i = gamma_ik * (x - mu) (x - mu)^T
			sigma.gemm(loggamma[k * N + i], xm, xm.T(), 1.0f);
      }

		sigma /= logGamma[k];

      _components[k].prepare();
   }
}



std::vector<int> GMMLayer::compute_labels(float *loggamma, int N, int K)
{
	std::vector<int> labels(N);

	for ( int i = 0; i < N; i++ )
	{
		int max_k = -1;
		float max_gamma = -INFINITY;

		for ( int k = 0; k < K; k++ )
		{
			if ( max_gamma < loggamma[k * N + i] )
			{
				max_k = k;
				max_gamma = loggamma[k * N + i];
			}
		}

		labels[i] = max_k;
	}

	return labels;
}



float GMMLayer::compute_entropy(float *loggamma, int N, const std::vector<int>& labels)
{
	float E = 0;

	for ( int i = 0; i < N; i++ )
	{
		int k = labels[i];

		E += log(loggamma[k * N + i]);
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
   float *logpi = new float[_K];
   float *loggamma = new float[_K * N];
   float *logGamma = new float[_K];

   for ( int k = 0; k < _K; k++ )
   {
      logpi[k] = log(_components[k].pi);
   }

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
         compute_log_likelihood_gamma_nk(logpi, _K, loggamma, N, &L);

         // check for convergence
         if ( fabs(L - L_prev) < TOLERANCE )
         {
				Logger::log(LogLevel::Debug, "converged after %d iterations", t + 1);
            break;
         }

         // M step
         // compute Gamma_k = sum(gamma_ik, i=1:N)
         compute_log_gamma_k(loggamma, N, _K, logGamma);

         float logGammaSum = compute_log_gamma_sum(logpi, _K, logGamma);

         // update parameters
         update(logpi, _K, loggamma, logGamma, logGammaSum, X);
      }

		// save outputs
		_log_likelihood = L;
		_num_parameters = _K * (1 + D + D * D);
		_num_samples = N;
		_labels = compute_labels(loggamma, N, _K);
		_entropy = compute_entropy(loggamma, N, _labels);
		_success = true;
	}
	catch ( std::runtime_error& e ) {
		_success = false;
	}

	Timer::pop();
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
