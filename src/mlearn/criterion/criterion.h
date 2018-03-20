/**
 * @file criterion/criterion.h
 *
 * Interface definitions for the abstract criterion layer.
 */
#ifndef CRITERION_H
#define CRITERION_H

#include <vector>
#include "mlearn/clustering/clustering.h"
#include "mlearn/util/iodevice.h"



namespace ML {



class CriterionLayer : public IODevice {
public:
	virtual ~CriterionLayer() {};
	virtual float compute(ClusteringLayer *layer) = 0;
};



}

#endif
