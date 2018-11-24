/**
 * @file feature/ica.h
 *
 * Interface definitions for the ICA feature layer.
 */
#ifndef MLEARN_FEATURE_ICA_H
#define MLEARN_FEATURE_ICA_H

#include "mlearn/layer/transformer.h"



namespace mlearn {



enum class ICANonl {
	none,
	pow3,
	tanh,
	gauss
};



class ICALayer : public TransformerLayer {
public:
	ICALayer(int n1, int n2, ICANonl nonl, int max_iter, float eps);
	ICALayer() : ICALayer(-1, -1, ICANonl::pow3, 1000, 0.0001f) {}

	void fit(const Matrix& X);
	void fit(const Matrix& X, const std::vector<int>& y, int c) { fit(X); }
	Matrix transform(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

private:
	Matrix fpica(const Matrix& X, const Matrix& W_z);

	int _n1;
	int _n2;
	ICANonl _nonl;
	int _max_iter;
	float _eps;
	Matrix _W;
};



}

#endif
