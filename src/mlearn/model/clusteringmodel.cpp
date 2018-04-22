/**
 * @file model/clusteringmodel.cpp
 *
 * Implementation of the clustering model.
 */
#include <iomanip>
#include "mlearn/model/clusteringmodel.h"
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace ML {



/**
 * Construct a clustering model.
 *
 * @param clustering
 * @param criterion
 */
ClusteringModel::ClusteringModel(const std::vector<ClusteringLayer *>& clustering, CriterionLayer *criterion)
{
	// initialize layers
	_clustering = clustering;
	_criterion = criterion;
	_best_layer = nullptr;

	// initialize stats
	_stats.error_rate = 0.0f;
	_stats.predict_time = 0.0f;
}



/**
 * Print information about a model.
 */
void ClusteringModel::print() const
{
	Logger::log(LogLevel::Verbose, "Hyperparameters");

	for ( ClusteringLayer *c : _clustering ) {
		c->print();
	}

	_criterion->print();

	Logger::log(LogLevel::Verbose, "");
}



/**
 * Perform clustering on the input data.
 *
 * @param X
 */
void ClusteringModel::fit(const std::vector<Matrix>& X)
{
	Timer::push("Clustering");

	// run clustering layers
	for ( ClusteringLayer *layer : _clustering ) {
		layer->fit(X);
	}

	// record prediction time
	_stats.predict_time = Timer::pop();

	// select model with lowest criterion value
	_best_layer = _criterion->select(_clustering);
}



/**
 * Score a model against ground truth labels.
 *
 * @param input
 * @param y_pred
 */
void ClusteringModel::score(const Dataset& input, const std::vector<int>& y_pred)
{
	// compute purity
	float purity = 0;

	int c = input.classes().size();
	int n = input.entries().size();
	int k = _best_layer->num_clusters();

	for ( int i = 0; i < k; i++ ) {
		int max_correct = 0;

		for ( int j = 0; j < c; j++ ) {
			int num_correct = 0;

			for ( int p = 0; p < n; p++ ) {
				if ( y_pred[p] == i && input.entries()[p].label == input.classes()[j] ) {
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
 * @oaram input
 * @param y_pred
 */
void ClusteringModel::print_results(const Dataset& input, const std::vector<int>& y_pred) const
{
	Logger::log(LogLevel::Verbose, "Results");

	for ( size_t i = 0; i < input.entries().size(); i++ ) {
		const DataEntry& entry = input.entries()[i];

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
		<< std::setw(12) << std::setprecision(3) << _stats.predict_time
		<< "\n";
}



}
