/**
 * @file feature/ica.h
 *
 * Interface definitions for the ICA feature layer.
 */
#ifndef ICA_H
#define ICA_H

#include "feature/feature.h"

namespace ML {

enum class ICANonl {
	none,
	pow3,
	tanh,
	gauss
};

class ICALayer : public FeatureLayer {
private:
	int n1;
	int n2;
	ICANonl nonl;
	int max_iter;
	float eps;

	Matrix fpica(const Matrix& X, const Matrix& W_z);

public:
	Matrix W;

	ICALayer(int n1, int n2, ICANonl nonl, int max_iter, float eps);
	ICALayer() : ICALayer(-1, -1, ICANonl::pow3, 1000, 0.0001f) {};

	void compute(const Matrix& X, const std::vector<DataEntry>& y, int c);
	Matrix project(const Matrix& X);

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print();
};

}

#endif
