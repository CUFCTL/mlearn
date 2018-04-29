/**
 * @file feature/ica.h
 *
 * Interface definitions for the ICA feature layer.
 */
#ifndef ML_FEATURE_ICA_H
#define ML_FEATURE_ICA_H

#include "mlearn/feature/feature.h"



namespace ML {



enum class ICANonl {
	none,
	pow3,
	tanh,
	gauss
};



class ICALayer : public FeatureLayer {
public:
	ICALayer(int n1, int n2, ICANonl nonl, int max_iter, float eps);
	ICALayer() : ICALayer(-1, -1, ICANonl::pow3, 1000, 0.0001f) {}

	void fit(const Matrix& X, const std::vector<int>& y, int c);
	Matrix transform(const Matrix& X);

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
