/**
 * @file model/clusteringmodel.cpp
 *
 * Implementation of the clustering model.
 */
#include <iomanip>
#include "model/clusteringmodel.h"
#include "util/logger.h"
#include "util/timer.h"

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
	this->_clustering = clustering;
	this->_criterion = criterion;
	this->_best_layer = nullptr;

	// initialize stats
	this->_stats.error_rate = 0.0f;
	this->_stats.predict_time = 0.0f;

	// log hyperparameters
	log(LL_VERBOSE, "Hyperparameters");

	for ( ClusteringLayer *c : this->_clustering ) {
		c->print();
	}

	log(LL_VERBOSE, "");
}

/**
 * Perform clustering on the input data.
 *
 * @param X
 */
void ClusteringModel::predict(const std::vector<Matrix>& X)
{
	timer_push("Clustering");

	// run clustering layers
	std::vector<int> results;

	for ( ClusteringLayer *c : this->_clustering ) {
		results.push_back(c->compute(X));
	}

	// record prediction time
	this->_stats.predict_time = timer_pop();

	// select model with lowest criterion value
	ClusteringLayer *min_c = nullptr;
	float min_value = 0;

	for ( size_t i = 0; i < this->_clustering.size(); i++ ) {
		if ( results[i] == 0 ) {
			ClusteringLayer *c = this->_clustering[i];
			float value = this->_criterion->compute(c);

			if ( min_c == nullptr || value < min_value ) {
				min_c = c;
				min_value = value;
			}

			log(LL_VERBOSE, "model %d: %8.3f", i, value);
		}
		else {
			log(LL_VERBOSE, "model %d: FAILED", i);
		}
	}
	log(LL_VERBOSE, "");

	if ( min_c == nullptr ) {
		log(LL_WARN, "warning: all models failed");
	}

	this->_best_layer = min_c;
}

/**
 * Validate a set of predicted labels against the ground truth.
 *
 * @param input
 * @param Y_pred
 */
void ClusteringModel::validate(const Dataset& input, const std::vector<int>& Y_pred)
{
	// compute purity
	float purity = 0;

	int c = input.labels().size();
	int n = input.entries().size();
	int k = this->_best_layer->num_clusters();

	for ( int i = 0; i < k; i++ ) {
		int max_correct = 0;

		for ( int j = 0; j < c; j++ ) {
			int num_correct = 0;

			for ( int p = 0; p < n; p++ ) {
				if ( Y_pred[p] == i && input.entries()[p].label == input.labels()[j] ) {
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
	this->_stats.error_rate = 1 - purity;
}

/**
 * Print prediction results of a model.
 *
 * @oaram input
 * @param Y_pred
 */
void ClusteringModel::print_results(const Dataset& input, const std::vector<int>& Y_pred) const
{
	log(LL_VERBOSE, "Results");

	for ( size_t i = 0; i < input.entries().size(); i++ ) {
		int y_pred = Y_pred[i];
		const DataEntry& entry = input.entries()[i];

		log(LL_VERBOSE, "%-4s (%s) -> %d",
			entry.name.c_str(),
			entry.label.c_str(),
			y_pred);
	}

	log(LL_VERBOSE, "Error rate: %.3f", this->_stats.error_rate);
	log(LL_VERBOSE, "");
}

/**
 * Print a model's performance and accuracy statistics.
 */
void ClusteringModel::print_stats() const
{
	std::cout
		<< std::setw(12) << std::setprecision(3) << this->_stats.error_rate
		<< std::setw(12) << std::setprecision(3) << this->_stats.predict_time
		<< "\n";
}

}
