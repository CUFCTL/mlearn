/**
 * @file layer/pipeline.cpp
 *
 * Implementation of the pipeline.
 */
#include <iomanip>
#include "mlearn/layer/pipeline.h"
#include "mlearn/util/iodevice.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace mlearn {



/**
 * Construct a pipeline.
 *
 * @param transforms
 * @param estimator
 */
Pipeline::Pipeline(std::vector<TransformerLayer *> transforms, EstimatorLayer *estimator):
	_transforms(transforms),
	_estimator(estimator)
{
}



/**
 * Save a pipeline to a file.
 *
 * @param file
 */
void Pipeline::save(IODevice& file) const
{
	for ( auto transform : _transforms )
	{
		file << *transform;
	}
	file << *_estimator;
}



/**
 * Load a pipeline from a file.
 *
 * @param file
 */
void Pipeline::load(IODevice& file)
{
	for ( auto transform : _transforms )
	{
		file >> *transform;
	}
	file >> *_estimator;
}



/**
 * Print information about a pipeline.
 */
void Pipeline::print() const
{
	Logger::log(LogLevel::Verbose, "Hyperparameters");

	for ( auto transform : _transforms )
	{
		transform->print();
	}
	_estimator->print();

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Fit the pipeline to a dataset.
 *
 * @param X
 */
void Pipeline::fit(const Matrix& X_)
{
	Timer::push("Training");

	// fit each transformer
	Matrix X(std::move(X_));

	for ( auto transform : _transforms )
	{
		transform->fit(X);
		X = transform->transform(X);

	}

	// fit estimator
	_estimator->fit(X);

	Timer::pop();
}



/**
 * Fit the pipeline to a dataset.
 *
 * @param X
 * @param y
 * @param c
 */
void Pipeline::fit(const Matrix& X_, const std::vector<int>& y, int c)
{
	Timer::push("Training");

	// fit each transformer
	Matrix X(std::move(X_));

	for ( auto transform : _transforms )
	{
		transform->fit(X, y, c);
		X = transform->transform(X);

	}

	// fit estimator
	_estimator->fit(X, y, c);

	Timer::pop();
}



/**
 * Use the pipeline to predict on a dataset.
 *
 * @param X
 */
std::vector<int> Pipeline::predict(const Matrix& X_) const
{
	Timer::push("Prediction");

	// perform feature extraction
	Matrix X(std::move(X_));

	for ( auto transform : _transforms )
	{
		X = transform->transform(X);
	}

	// compute predicted labels
	std::vector<int> y_pred = _estimator->predict(X);

	Timer::pop();

	return y_pred;
}



/**
 * Score a pipeline against ground truth labels.
 *
 * @param X
 * @param y
 */
float Pipeline::score(const Matrix& X_, const std::vector<int>& y) const
{
	// perform feature extraction
	Matrix X(std::move(X_));

	for ( auto transform : _transforms )
	{
		X = transform->transform(X);
	}

	// score estimator
	return _estimator->score(X, y);
}



}
