/**
 * @file layer/estimator.cpp
 *
 * Implementation of the abstract estimator layer.
 */
#include "mlearn/layer/estimator.h"



namespace mlearn {



/**
 * Score an estimator against ground truth labels.
 *
 * @param X
 * @param y
 */
float EstimatorLayer::score(const Matrix& X, const std::vector<int>& y) const
{
	// compute predicted labels
	std::vector<int> y_pred = predict(X);

	// compute accuracy of labels against ground truth
	int num_correct = 0;

	for ( size_t i = 0; i < y.size(); i++ )
	{
		if ( y_pred[i] == y[i] )
		{
			num_correct++;
		}
	}

	return (float) num_correct / y.size();
}



}
