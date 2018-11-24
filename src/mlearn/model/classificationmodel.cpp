/**
 * @file model/classificationmodel.cpp
 *
 * Implementation of the classification model.
 */
#include <iomanip>
#include "mlearn/model/classificationmodel.h"
#include "mlearn/util/iodevice.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace mlearn {



/**
 * Construct a classification model.
 *
 * @param feature
 * @param classifier
 */
ClassificationModel::ClassificationModel(FeatureLayer *feature, ClassifierLayer *classifier)
{
	// initialize layers
	_scaler = Scaler(true, false);
	_feature = feature;
	_classifier = classifier;
}



/**
 * Save a model to a file.
 *
 * @param path
 */
void ClassificationModel::save(const std::string& path)
{
	IODevice file(path, std::ofstream::out);

	file << _scaler;
	if ( _feature ) file << *_feature;
	file << *_classifier;
}



/**
 * Load a model from a file.
 *
 * @param path
 */
void ClassificationModel::load(const std::string& path)
{
	IODevice file(path, std::ifstream::in);

	file >> _scaler;
	if ( _feature ) file >> *_feature;
	file >> *_classifier;
}



/**
 * Print information about a model.
 */
void ClassificationModel::print() const
{
	Logger::log(LogLevel::Verbose, "Hyperparameters");

	if ( _feature ) { _feature->print(); }
	_classifier->print();

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Fit the model to a training set.
 *
 * @param X
 * @param y
 * @param c
 */
void ClassificationModel::fit(const Matrix& X_, const std::vector<int>& y, int c)
{
	Timer::push("Training");

	// scale data
	_scaler.fit(X_);
	Matrix X = _scaler.transform(X_);

	// perform feature extraction
	if ( _feature )
	{
		_feature->fit(X, y, c);
		X = _feature->transform(X);
	}

	// fit classifier
	_classifier->fit(X, y, c);

	Timer::pop();
}



/**
 * Use the model to predict on a dataset.
 *
 * @param X
 */
std::vector<int> ClassificationModel::predict(const Matrix& X_) const
{
	Timer::push("Prediction");

	// scale data
	Matrix X = _scaler.transform(X_);

	// perform feature extraction
	if ( _feature )
	{
		X = _feature->transform(X);
	}

	// compute predicted labels
	std::vector<int> y_pred = _classifier->predict(X);

	Timer::pop();

	return y_pred;
}



/**
 * Score a model against ground truth labels.
 *
 * @param y_true
 * @param y_pred
 */
float ClassificationModel::score(const std::vector<int>& y_true, const std::vector<int>& y_pred) const
{
	int num_errors = 0;

	for ( size_t i = 0; i < y_true.size(); i++ ) {
		if ( y_pred[i] != y_true[i] ) {
			num_errors++;
		}
	}

	return (float) num_errors / y_true.size();
}



}
