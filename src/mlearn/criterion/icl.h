/**
 * @file clustering/icl.h
 *
 * Interface definitions for the ICL layer.
 */
#ifndef ML_CRITERION_ICL_H
#define ML_CRITERION_ICL_H

#include "mlearn/criterion/criterion.h"



namespace ML {



class ICLLayer : public CriterionLayer {
public:
	float score(ClusteringLayer *layer);
	void print() const;
};



}

#endif
