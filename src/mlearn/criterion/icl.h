/**
 * @file clustering/icl.h
 *
 * Interface definitions for the ICL layer.
 */
#ifndef MLEARN_CRITERION_ICL_H
#define MLEARN_CRITERION_ICL_H

#include "mlearn/criterion/criterion.h"



namespace mlearn {



class ICLLayer : public CriterionLayer {
public:
	float score(ClusteringLayer *layer);
	void print() const;
};



}

#endif
