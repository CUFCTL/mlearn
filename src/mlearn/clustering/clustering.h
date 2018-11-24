/**
 * @file clustering/clustering.h
 *
 * Interface definitions for the abstract clustering layer.
 */
#ifndef MLEARN_CLUSTERING_CLUSTERING_H
#define MLEARN_CLUSTERING_CLUSTERING_H

#include <vector>
#include "mlearn/layer/layer.h"
#include "mlearn/math/matrix.h"



namespace mlearn {



class ClusteringLayer : public Layer {
public:
	virtual ~ClusteringLayer() {}
	virtual void fit(const std::vector<Matrix>& X) = 0;
	virtual std::vector<int> predict(const std::vector<Matrix>& X) = 0;

	virtual float entropy() const = 0;
	virtual float log_likelihood() const = 0;
	virtual int num_clusters() const = 0;
	virtual int num_parameters() const = 0;
	virtual int num_samples() const = 0;
	virtual bool success() const = 0;
};



}

#endif
