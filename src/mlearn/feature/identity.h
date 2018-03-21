/**
 * @file feature/identity.h
 *
 * Interface definitions for the identity feature layer.
 */
#ifndef IDENTITY_H
#define IDENTITY_H

#include "mlearn/feature/feature.h"



namespace ML {



class IdentityLayer : public FeatureLayer {
public:
	void compute(const Matrix& X, const std::vector<int>& y, int c) {}
	Matrix project(const Matrix& X);

	void save(IODevice& file) const {}
	void load(IODevice& file) {}
	void print() const;
};



}

#endif
