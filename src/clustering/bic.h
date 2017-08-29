/**
 * @file clustering/bic.h
 *
 * Interface definitions for the BIC layer.
 */
#ifndef BIC_H
#define BIC_H

#include <vector>
#include "clustering/criterion.h"

namespace ML {

class BICLayer : public CriterionLayer {
public:
	float compute(ClusteringLayer *layer);
};

}

#endif
