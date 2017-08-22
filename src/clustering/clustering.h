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

	virtual void compute(const Matrix& X) = 0;

	virtual std::vector<int> output() const = 0;
	virtual precision_t error() const = 0;

	virtual void print() const = 0;
};

}

#endif
