/**
 * @file classifier/bayes.cpp
 *
 * Implementation of the naive Bayes classifier.
 */
#include <algorithm>
#include "mlearn/classifier/bayes.h"
#include "mlearn/feature/lda.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/util/logger.h"



namespace ML {



/**
 * Compute intermediate data for classification.
 *
 * @param X
 * @param y
 * @param c
 */
void BayesLayer::fit(const Matrix& X, const std::vector<int>& y, int c)
{
	std::vector<Matrix> X_c = m_copy_classes(X, y, c);

	// compute class means
	_mu = m_class_means(X_c);

	// compute class covariances
	std::vector<Matrix> S = m_class_scatters(X_c, _mu);

	// compute inverses of each class covariance
	_S_inv.reserve(c);

	for ( size_t i = 0; i < c; i++ ) {
		_S_inv.push_back(S[i].inverse());
	}
}



/**
 * Compute the probability of a class for a
 * feature vector using the Bayes discriminant
 * function:
 *
 *   g_i'(x) = -1/2 * (x - mu_i)' * S_i^-1 * (x - mu_i)
 *
 * @param x
 * @param mu
 * @param S_inv
 */
float BayesLayer::prob(Matrix x, const Matrix& mu, const Matrix& S_inv)
{
	x -= mu;

	return -0.5f * (x.T() * S_inv).dot(x);
}



/**
 * Classify an observation using naive Bayes.
 *
 * @param X_test
 * @return predicted labels of the test observations
 */
std::vector<int> BayesLayer::predict(const Matrix& X_test)
{
	// compute label for each test vector
	std::vector<int> y_pred(X_test.cols());

	for ( int i = 0; i < X_test.cols(); i++ ) {
		std::vector<float> probs(_mu.size());

		// compute the Bayes probability for each class
		for ( int j = 0; j < probs.size(); j++ ) {
			probs[j] = prob(X_test(i), _mu[j], _S_inv[j]);
		}

		// select the class with the highest probability
		y_pred[i] = max_element(probs.begin(), probs.end()) - probs.begin();
	}

	return y_pred;
}



/**
 * Save a Bayes layer to a file.
 *
 * @param file
 */
void BayesLayer::save(IODevice& file) const
{
	file << _mu;
	file << _S_inv;
}



/**
 * Load a Bayes layer from a file.
 *
 * @param file
 */
void BayesLayer::load(IODevice& file)
{
	file >> _mu;
	file >> _S_inv;
}



/**
 * Print information about a Bayes classifier.
 */
void BayesLayer::print() const
{
	Logger::log(LogLevel::Verbose, "Bayes");
}



}
