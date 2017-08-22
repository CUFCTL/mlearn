/**
 * @file clustering/clustering.h
 *
 * Interface definitions for the abstract clustering layer.
 */
#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include "math/matrix.h"

namespace ML {

class ClusteringLayer {
public:
	virtual ~ClusteringLayer() {};

	virtual std::vector<int> compute(const Matrix& X, int k) = 0;
};

}

#endif
