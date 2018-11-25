/**
 * @file clustering/clustering.cpp
 *
 * Implementation of the abstract clustering layer.
 */
#include "mlearn/clustering/clustering.h"
#include <algorithm>



namespace mlearn {



/**
 * Score a model against ground truth labels.
 *
 * @param X
 * @param y
 */
float ClusteringLayer::score(const Matrix& X, const std::vector<int>& y) const
{
	// compute predicted labels
	std::vector<int> y_pred = predict(X);

	// compute purity of labels against ground truth
	float purity = 0;

	int c = 1 + *std::max_element(y.begin(), y.end());
	int k = num_clusters();

	for ( int i = 0; i < k; i++ )
	{
		int max_correct = 0;

		for ( int j = 0; j < c; j++ )
		{
			int num_correct = 0;

			for ( int p = 0; p < y.size(); p++ )
			{
				if ( y_pred[p] == i && y[p] == j )
				{
					num_correct++;
				}
			}

			if ( max_correct < num_correct )
			{
				max_correct = num_correct;
			}
		}

		purity += max_correct;
	}

	purity /= y.size();

	return purity;
}



}
