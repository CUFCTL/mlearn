/**
 * @file criterion/criterion.h
 *
 * Interface definitions for the abstract criterion layer.
 */
#ifndef ML_CRITERION_CRITERION_H
#define ML_CRITERION_CRITERION_H

#include <vector>
#include "mlearn/clustering/clustering.h"
#include "mlearn/util/iodevice.h"



namespace ML {



class CriterionLayer {
public:
	virtual ~CriterionLayer() {};
	virtual ClusteringLayer * select(const std::vector<ClusteringLayer *>& layers);
	virtual float score(ClusteringLayer *layer) = 0;
	virtual void print() const = 0;
};



}

#endif
