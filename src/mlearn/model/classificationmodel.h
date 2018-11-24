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

	const Dataset& train_set() const { return _train_set; }

	void save(const std::string& path);
	void load(const std::string& path);
	void print() const;

	void fit(const Dataset& dataset);
	std::vector<int> predict(const Dataset& dataset) const;
	float score(const Dataset& dataset, const std::vector<int>& y_pred) const;

	void print_results(const Dataset& dataset, const std::vector<int>& y_pred) const;

private:
	// layers
	Dataset _train_set;
	Scaler _scaler;
	FeatureLayer *_feature;
	ClassifierLayer *_classifier;
};



}

#endif
