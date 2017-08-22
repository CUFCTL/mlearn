/**
 * @file clustering/kmeans.h
 *
 * Interface definitions for the k-means clustering layer.
 */
#ifndef KMEANS_H
#define KMEANS_H

#include "clustering/clustering.h"

namespace ML {

class KMeansLayer : public ClusteringLayer {
public:
	std::vector<int> compute(const Matrix& X, int k);
};

}

#endif
