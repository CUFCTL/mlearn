/**
 * @file criterion/bic.h
 *
 * Interface definitions for the BIC layer.
 */
#ifndef BIC_H
#define BIC_H

#include "mlearn/criterion/criterion.h"



namespace ML {



class BICLayer : public CriterionLayer {
public:
	float compute(ClusteringLayer *layer);

	void print() const;
};



}

#endif
