/**
 * @file preprocessing/scaler.h
 *
 * Interface definitions for the scaler type.
 */
#ifndef MLEARN_PREPROCESSING_SCALER_H
#define MLEARN_PREPROCESSING_SCALER_H

#include "mlearn/layer/layer.h"
#include "mlearn/math/matrix.h"


namespace mlearn {



class Scaler : public Layer {
public:
	Scaler(bool with_mean=true, bool with_std=true);

	void fit(const Matrix& X);
	void transform(Matrix& X);

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const {}

private:
	bool _with_mean;
	bool _with_std;
	Matrix _mean;
	Matrix _std;
};



}

#endif
