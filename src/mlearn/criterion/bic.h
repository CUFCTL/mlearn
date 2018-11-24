/**
 * @file criterion/bic.h
 *
 * Interface definitions for the BIC layer.
 */
#ifndef MLEARN_CRITERION_BIC_H
#define MLEARN_CRITERION_BIC_H

#include "mlearn/criterion/criterion.h"



namespace mlearn {



class BICLayer : public CriterionLayer {
public:
	float score(ClusteringLayer *layer);
	void print() const;
};



}

#endif
