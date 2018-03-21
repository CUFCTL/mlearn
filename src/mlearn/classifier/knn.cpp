/**
 * @file classifier/knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <algorithm>
#include "mlearn/classifier/knn.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/util/logger.h"



namespace ML {



typedef float (*dist_func_t)(const Matrix&, int, const Matrix&, int);



typedef struct {
	int label;
	float dist;
} neighbor_t;



typedef struct {
	int id;
	int count;
} item_count_t;



/**
 * Comparison function for sorting neighbors.
 *
 * @param a
 * @param b
 */
bool kNN_compare(const neighbor_t& a, const neighbor_t& b)
{
	return (a.dist < b.dist);
}



/**
 * Determine the mode of a list of neighbors.
 *
 * @param items
 */
int kNN_mode(const std::vector<neighbor_t>& items)
{
	std::vector<item_count_t> counts;

	// compute the frequency of each item in the list
	for ( const neighbor_t& item : items ) {
		int id = item.label;

		size_t j = 0;
		while ( j < counts.size() && counts[j].id != id ) {
			j++;
		}

		if ( j == counts.size() ) {
			item_count_t count;
			count.id = id;
			count.count = 1;

			counts.push_back(count);
		}
		else {
			counts[j].count++;
		}
	}

	// find the item with the highest frequency
	item_count_t max = counts[0];

	for ( size_t i = 1; i < counts.size(); i++ ) {
		if ( max.count < counts[i].count ) {
			max = counts[i];
		}
	}

	return max.id;
}



/**
 * Construct a kNN classifier.
 *
 * @param k
 * @param dist
 */
KNNLayer::KNNLayer(int k, KNNDist dist)
{
	_k = k;
	_dist = dist;
}



/**
 * Compute intermediate data for classification.
 *
 * @param X
 * @param y
 * @param c
 */
void KNNLayer::compute(const Matrix& X, const std::vector<int>& y, int c)
{
	_X = X;
	_y = y;
}



/**
 * Classify an observation using k-nearest neighbors.
 *
 * @param X_test
 */
std::vector<int> KNNLayer::predict(const Matrix& X_test)
{
	// determine distance function
	dist_func_t dist = nullptr;

	if ( _dist == KNNDist::COS ) {
		dist = m_dist_COS;
	}
	else if ( _dist == KNNDist::L1 ) {
		dist = m_dist_L1;
	}
	else if ( _dist == KNNDist::L2 ) {
		dist = m_dist_L2;
	}

	std::vector<int> y_pred(X_test.cols());

	for ( int i = 0; i < X_test.cols(); i++ ) {
		// compute distance between X_test_i and each X_i
		std::vector<neighbor_t> neighbors;
		neighbors.reserve(_X.cols());

		for ( int j = 0; j < _X.cols(); j++ ) {
			neighbor_t n;
			n.label = _y[j];
			n.dist = dist(X_test, i, _X, j);

			neighbors.push_back(n);
		}

		// determine the k nearest neighbors
		std::sort(neighbors.begin(), neighbors.end(), kNN_compare);

		neighbors.erase(neighbors.begin() + _k, neighbors.end());

		// determine the mode of the k nearest labels
		y_pred[i] = kNN_mode(neighbors);
	}

	return y_pred;
}



/**
 * Save a KNN layer to a file.
 *
 * @param file
 */
void KNNLayer::save(IODevice& file) const
{
	file << _k;
	file << (int) _dist;
	file << _X;
	file << _y;
}



/**
 * Load a KNN layer from a file.
 *
 * @param file
 */
void KNNLayer::load(IODevice& file)
{
	file >> _k;
	int dist; file >> dist; _dist = (KNNDist) dist;
	file >> _X;
	file >> _y;
}



/**
 * Print information about a kNN classifier.
 */
void KNNLayer::print() const
{
	const char *dist_name = "";

	if ( _dist == KNNDist::COS ) {
		dist_name = "COS";
	}
	else if ( _dist == KNNDist::L1 ) {
		dist_name = "L1";
	}
	else if ( _dist == KNNDist::L2 ) {
		dist_name = "L2";
	}

	Logger::log(LogLevel::Verbose, "kNN");
	Logger::log(LogLevel::Verbose, "  %-20s  %10d", "k", _k);
	Logger::log(LogLevel::Verbose, "  %-20s  %10s", "dist", dist_name);
}



}
