/**
 * @file model/classificationmodel.h
 *
 * Interface definitions for the classification model.
 */
#ifndef MLEARN_MODEL_CLASSIFICATIONMODEL_H
#define MLEARN_MODEL_CLASSIFICATIONMODEL_H

#include "mlearn/classifier/classifier.h"
#include "mlearn/data/dataset.h"
#include "mlearn/feature/feature.h"
#include "mlearn/preprocessing/scaler.h"



namespace mlearn {



class ClassificationModel {
public:
	ClassificationModel(FeatureLayer *feature, ClassifierLayer *classifier);
	~ClassificationModel() {}

	void save(const std::string& path);
	void load(const std::string& path);
	void print() const;

	void fit(const Matrix& X, const std::vector<int>& y, int c);
	std::vector<int> predict(const Matrix& X) const;
	float score(const std::vector<int>& y_true, const std::vector<int>& y_pred) const;

private:
	// layers
	Scaler _scaler;
	FeatureLayer *_feature;
	ClassifierLayer *_classifier;
};



}

#endif
