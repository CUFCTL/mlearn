/**
 * @file feature/lda.h
 *
 * Interface definitions for the LDA feature layer.
 */
#ifndef MLEARN_FEATURE_LDA_H
#define MLEARN_FEATURE_LDA_H

#include "mlearn/layer/transformer.h"



namespace mlearn {



class LDALayer : public TransformerLayer {
public:
	LDALayer(int n1, int n2);
	LDALayer() : LDALayer(-1, -1) {}

	void fit(const Matrix& X) {}
	void fit(const Matrix& X, const std::vector<int>& y, int c);
	Matrix transform(const Matrix& X) const;

	void save(IODevice& file) const;
	void load(IODevice& file);
	void print() const;

private:
	int _n1;
	int _n2;
	Matrix _W;
};



}

#endif
