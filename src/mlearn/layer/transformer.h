/**
 * @file layer/transformer.h
 *
 * Interface definitions for the abstract transformer layer.
 */
#ifndef MLEARN_LAYER_TRANSFORMER_H
#define MLEARN_LAYER_TRANSFORMER_H

#include "mlearn/layer/layer.h"
#include "mlearn/math/matrix.h"



namespace mlearn {



class TransformerLayer : public Layer {
public:
	virtual ~TransformerLayer() {};

	virtual void fit(const Matrix& X) = 0;
	virtual void fit(const Matrix& X, const std::vector<int>& y, int c) = 0;
	virtual Matrix transform(const Matrix& X) const = 0;
};



}

#endif
