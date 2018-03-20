/**
 * @file feature/pca.h
 *
 * Interface definitions for the PCA feature layer.
 */
#ifndef PCA_H
#define PCA_H

#include "mlearn/feature/feature.h"



namespace ML {



class PCALayer : public FeatureLayer {
public:
	PCALayer(int n1);
	PCALayer() : PCALayer(-1) {}

	const Matrix& W() const { return _W; }
	const Matrix& D() const { return _D; }

	void compute(const Matrix& X, const std::vector<int>& y, int c);
	Matrix project(const Matrix& X);

	void save(std::ofstream& file);
	void load(std::ifstream& file);
	void print() const;

private:
	int _n1;
	Matrix _W;
	Matrix _D;
};



}

#endif
