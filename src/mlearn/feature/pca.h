/**
 * @file feature/pca.h
 *
 * Interface definitions for the PCA feature layer.
 */
#ifndef MLEARN_FEATURE_PCA_H
#define MLEARN_FEATURE_PCA_H

#include "mlearn/layer/transformer.h"



namespace mlearn {



class PCALayer : public TransformerLayer {
public:
	PCALayer(int n1);
	PCALayer() : PCALayer(-1) {}

	const Matrix& W() const { return _W; }
	const Matrix& D() const { return _D; }

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c) { fit(X); }
	Matrix transform(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

private:
	int _n1;
	Matrix _W;
	Matrix _D;
};



}

#endif
