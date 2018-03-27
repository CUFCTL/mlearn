/**
 * @file model/classificationmodel.h
 *
 * Interface definitions for the classification model.
 */
#ifndef CLASSIFICATIONMODEL_H
#define CLASSIFICATIONMODEL_H

#include "mlearn/classifier/classifier.h"
#include "mlearn/data/dataset.h"
#include "mlearn/feature/feature.h"



namespace ML {



class ClassificationModel {
public:
	ClassificationModel(FeatureLayer *feature, ClassifierLayer *classifier);
	~ClassificationModel() {}

	const Dataset& train_set() const { return _train_set; }

	void save(const std::string& path);
	void load(const std::string& path);
	void print() const;

	void fit(const Dataset& dataset);
	std::vector<int> predict(const Dataset& dataset);
	void validate(const Dataset& dataset, const std::vector<int>& y_pred);

	void print_results(const Dataset& dataset, const std::vector<int>& y_pred) const;
	void print_stats() const;

private:
	// input data
	Dataset _train_set;
	Matrix _mean;

	// feature layer
	FeatureLayer *_feature;

	// classifier layer
	ClassifierLayer *_classifier;

	// performance, accuracy stats
	struct {
		float error_rate;
		float fit_time;
		float predict_time;
	} _stats;
};



}

#endif
