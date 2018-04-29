/**
 * @file criterion/bic.h
 *
 * Interface definitions for the BIC layer.
 */
#ifndef ML_CRITERION_BIC_H
#define ML_CRITERION_BIC_H

#include "mlearn/criterion/criterion.h"



namespace ML {



class BICLayer : public CriterionLayer {
public:
	float score(ClusteringLayer *layer);
	void print() const;
};



}

#endif
