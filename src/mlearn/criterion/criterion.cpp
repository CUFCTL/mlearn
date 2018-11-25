/**
 * @file criterion/criterion.cpp
 *
 * Implementation of the criterion layer.
 */
#include "mlearn/criterion/criterion.h"
#include "mlearn/util/logger.h"



namespace mlearn {



CriterionLayer::CriterionLayer(Criterion criterion, const std::vector<ClusteringLayer*>& models):
	_criterion(criterion),
	_models(models)
{
}



/**
 * Save a criterion layer to a file.
 *
 * @param file
 */
void CriterionLayer::save(IODevice& file) const
{
	file << (int) _criterion;

	for ( auto model : _models )
	{
		file << *model;
	}
}



/**
 * Load a criterion layer from a file.
 *
 * @param file
 */
void CriterionLayer::load(IODevice& file)
{
	int criterion; file >> criterion; _criterion = (Criterion) criterion;

	for ( auto model : _models )
	{
		file >> *model;
	}
}



/**
 * Print information about a criterion layer.
 */
void CriterionLayer::print() const
{
	const char *criterion_name = "";

	if ( _criterion == Criterion::AIC )
	{
		criterion_name = "AIC";
	}
	else if ( _criterion == Criterion::BIC )
	{
		criterion_name = "BIC";
	}
	else if ( _criterion == Criterion::ICL )
	{
		criterion_name = "ICL";
	}

	Logger::log(LogLevel::Verbose, "Criterion (%s)", criterion_name);

	for ( auto model : _models ) {
		model->print();
	}

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Fit model to a dataset.
 *
 * @param X
 */
void CriterionLayer::fit(const Matrix& X)
{
	float min_value = INFINITY;

	for ( size_t i = 0; i < _models.size(); i++ )
	{
		// fit clustering model
		_models[i]->fit(X);

		// score clustering model with a criterion
		float value;

		switch ( _criterion ) {
		case Criterion::AIC:
			value = _models[i]->aic();
			break;
		case Criterion::BIC:
			value = _models[i]->bic();
			break;
		case Criterion::ICL:
			value = _models[i]->icl();
			break;
		}

		// select model with lowest criterion value
		if ( value < min_value )
		{
			_selected_model = _models[i];
			min_value = value;
		}

		Logger::log(LogLevel::Verbose, "model %d: %8.3f", i + 1, value);
	}
	Logger::log(LogLevel::Verbose, "");

	if ( _selected_model == nullptr )
	{
		Logger::log(LogLevel::Warn, "warning: all models failed");
	}
}



/**
 * Predict labels for a dataset.
 *
 * @param X
 */
std::vector<int> CriterionLayer::predict(const Matrix& X) const
{
	return _selected_model->predict(X);
}



/**
 * Score a model against ground truth labels.
 *
 * @param X
 * @param y
 */
float CriterionLayer::score(const Matrix& X, const std::vector<int>& y) const
{
	return _selected_model->score(X, y);
}



}
