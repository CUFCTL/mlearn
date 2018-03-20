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
#include "mlearn/math/matrix.h"

namespace ML {

class ClassificationModel {
private:
	// input data
	Dataset _train_set;
	Matrix _mean;

	// feature layer
	FeatureLayer *_feature;
	Matrix _P;

	// classifier layer
	ClassifierLayer *_classifier;

	// performance, accuracy stats
	struct {
		float error_rate;
		float train_time;
		float predict_time;
	} _stats;

public:
	ClassificationModel(FeatureLayer *feature, ClassifierLayer *classifier);
	~ClassificationModel() {};

	void save(const std::string& path);
	void load(const std::string& path);

	void train(const Dataset& train_set);
	std::vector<int> predict(const Dataset& test_set);
	void validate(const Dataset& test_set, const std::vector<int>& y_pred);

	void print_results(const Dataset& test_set, const std::vector<int>& y_pred) const;
	void print_stats() const;
};

}

#endif
