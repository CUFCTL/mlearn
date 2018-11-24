/**
 * @file model/clusteringmodel.cpp
 *
 * Implementation of the clustering model.
 */
#include <iomanip>
#include "mlearn/model/clusteringmodel.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace mlearn {



/**
 * Construct a clustering model.
 *
 * @param models
 * @param criterion
 * @param clustering
 */
ClusteringModel::ClusteringModel(const std::vector<ClusteringLayer *>& models, CriterionLayer *criterion, ClusteringLayer *clustering)
{
	// initialize layers
	_models = models;
	_criterion = criterion;
	_clustering = clustering;

	// initialize stats
	_stats.error_rate = 0.0f;
	_stats.fit_time = 0.0f;
	_stats.predict_time = 0.0f;
}



/**
 * Save a model to a file.
 *
 * @param path
 */
void ClusteringModel::save(const std::string& path)
{
	IODevice file(path, std::ofstream::out);

	file << *_clustering;
}



/**
 * Load a model from a file.
 *
 * @param path
 */
void ClusteringModel::load(const std::string& path)
{
	IODevice file(path, std::ifstream::in);

	file >> *_clustering;
}



/**
 * Print information about a model.
 */
void ClusteringModel::print() const
{
	Logger::log(LogLevel::Verbose, "Hyperparameters");

	for ( ClusteringLayer *c : _models ) {
		c->print();
	}

	_criterion->print();

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Fit model to a dataset.
 *
 * @param X
 */
void ClusteringModel::fit(const std::vector<Matrix>& X)
{
	Timer::push("Clustering");

	// run clustering layers
	for ( ClusteringLayer *layer : _models ) {
		layer->fit(X);
	}

	// record fit time
	_stats.fit_time = Timer::pop();

	// select model with lowest criterion value
	_clustering = _criterion->select(_models);
}



/**
 * Predict labels for a dataset.
 *
 * @param X
 */
std::vector<int> ClusteringModel::predict(const std::vector<Matrix>& X)
{
	Timer::push("Prediction");

	// predict labels
	std::vector<int> y_pred = _clustering->predict(X);

	// record prediction time
	_stats.predict_time = Timer::pop();

	return y_pred;
}



/**
 * Score a model against ground truth labels.
 *
 * @param dataset
 * @param y_pred
 */
void ClusteringModel::score(const Dataset& dataset, const std::vector<int>& y_pred)
{
	// compute purity
	float purity = 0;

	int c = dataset.classes().size();
	int n = dataset.entries().size();
	int k = _clustering->num_clusters();

	for ( int i = 0; i < k; i++ ) {
		int max_correct = 0;

		for ( int j = 0; j < c; j++ ) {
			int num_correct = 0;

			for ( int p = 0; p < n; p++ ) {
				if ( y_pred[p] == i && dataset.entries()[p].label == dataset.classes()[j] ) {
					num_correct++;
				}
			}

			if ( max_correct < num_correct ) {
				max_correct = num_correct;
			}
		}

		purity += max_correct;
	}

	purity /= n;

	// compute error rate
	_stats.error_rate = 1 - purity;
}



/**
 * Print prediction results of a model.
 *
 * @oaram dataset
 * @param y_pred
 */
void ClusteringModel::print_results(const Dataset& dataset, const std::vector<int>& y_pred) const
{
	Logger::log(LogLevel::Verbose, "Results");

	for ( size_t i = 0; i < dataset.entries().size(); i++ ) {
		const DataEntry& entry = dataset.entries()[i];

		Logger::log(LogLevel::Verbose, "%-4s (%s) -> %d",
			entry.name.c_str(),
			entry.label.c_str(),
			y_pred[i]);
	}

	Logger::log(LogLevel::Verbose, "Error rate: %.3f", _stats.error_rate);
	Logger::log(LogLevel::Verbose, "");
}



/**
 * Print a model's performance and accuracy statistics.
 */
void ClusteringModel::print_stats() const
{
	std::cout
		<< std::setw(12) << std::setprecision(3) << _stats.error_rate
		<< std::setw(12) << std::setprecision(3) << _stats.fit_time
		<< std::setw(12) << std::setprecision(3) << _stats.predict_time
		<< "\n";
}



}
