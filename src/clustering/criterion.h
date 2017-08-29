/**
 * @file clustering/criterion.h
 *
 * Interface definitions for the abstract criterion layer.
 */
#ifndef CRITERION_H
#define CRITERION_H

#include <vector>
#include "clustering/clustering.h"

namespace ML {

class CriterionLayer {
public:
	virtual ~CriterionLayer() {};

	virtual float compute(ClusteringLayer *layer) = 0;
};

}

#endif
