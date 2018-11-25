/**
 * @file clustering/clustering.h
 *
 * Interface definitions for the abstract clustering layer.
 */
#ifndef MLEARN_CLUSTERING_CLUSTERING_H
#define MLEARN_CLUSTERING_CLUSTERING_H

#include <vector>
#include "mlearn/layer/estimator.h"
#include "mlearn/math/matrix.h"



namespace mlearn {



class ClusteringLayer : public EstimatorLayer {
public:
	virtual ~ClusteringLayer() {}

	float score(const Matrix& X, const std::vector<int>& y) const;

	virtual int num_clusters() const = 0;
	virtual float aic() const = 0;
	virtual float bic() const = 0;
	virtual float icl() const = 0;
};



}

#endif
