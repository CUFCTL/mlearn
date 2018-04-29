/**
 * @file clustering/clustering.h
 *
 * Interface definitions for the abstract clustering layer.
 */
#ifndef ML_CLUSTERING_CLUSTERING_H
#define ML_CLUSTERING_CLUSTERING_H

#include <vector>
#include "mlearn/math/matrix.h"
#include "mlearn/util/iodevice.h"



namespace ML {



class ClusteringLayer {
public:
	virtual ~ClusteringLayer() {};
	virtual void fit(const std::vector<Matrix>& X) = 0;
	virtual std::vector<int> predict(const std::vector<Matrix>& X) = 0;
	virtual void print() const = 0;

	virtual float entropy() const = 0;
	virtual float log_likelihood() const = 0;
	virtual int num_clusters() const = 0;
	virtual int num_parameters() const = 0;
	virtual int num_samples() const = 0;
	virtual bool success() const = 0;
};



}

#endif
