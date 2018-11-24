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

	file << _train_set;
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

	file >> _train_set;
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
 * @param dataset
 */
void ClassificationModel::fit(const Dataset& dataset)
{
	Timer::push("Training");

	_train_set = dataset;

	Logger::log(LogLevel::Verbose, "Training set: %d samples, %d classes",
		dataset.entries().size(),
		dataset.classes().size());

	// load data
	Matrix X = dataset.load_data();

	// scale data
	_scaler.fit(X);
	X = _scaler.transform(X);

	// perform feature extraction
	if ( _feature )
	{
		_feature->fit(X, _train_set.labels(), _train_set.classes().size());
		X = _feature->transform(X);
	}

	// fit classifier
	_classifier->fit(X, _train_set.labels(), _train_set.classes().size());

	Timer::pop();

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Use the model to predict on a dataset.
 *
 * @param dataset
 */
std::vector<int> ClassificationModel::predict(const Dataset& dataset) const
{
	Timer::push("Prediction");

	Logger::log(LogLevel::Verbose, "Test set: %d samples, %d classes",
		dataset.entries().size(),
		dataset.classes().size());

	// load data
	Matrix X = dataset.load_data();

	// scale data
	X = _scaler.transform(X);

	// perform feature extraction
	if ( _feature )
	{
		X = _feature->transform(X);
	}

	// compute predicted labels
	std::vector<int> y_pred = _classifier->predict(X);

	Timer::pop();

	Logger::log(LogLevel::Verbose, "");

	return y_pred;
}



/**
 * Score a model against ground truth labels.
 *
 * @param dataset
 * @param y_pred
 */
float ClassificationModel::score(const Dataset& dataset, const std::vector<int>& y_pred) const
{
	int num_errors = 0;

	for ( size_t i = 0; i < dataset.labels().size(); i++ ) {
		if ( y_pred[i] != dataset.labels()[i] ) {
			num_errors++;
		}
	}

	return (float) num_errors / dataset.entries().size();
}



}
