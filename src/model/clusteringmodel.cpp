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
 * @param input
 */
std::vector<int> ClusteringModel::predict(const Dataset& input)
{
	timer_push("Load data");

	this->_input = input;

	Matrix X = this->_input.load_data();

	timer_pop();

	timer_push("Clustering");

	// run clustering layers
	for ( ClusteringLayer *c : this->_clustering ) {
		c->compute(X);
	}

	// record prediction time
	this->_stats.predict_time = timer_pop();

	// select model with lowest criterion value
	ClusteringLayer *min_c = nullptr;
	float min_value = 0;

	for ( ClusteringLayer *c : this->_clustering ) {
		float value = this->_criterion->compute(c);

		if ( min_c == nullptr || value < min_value ) {
			min_c = c;
			min_value = value;
		}

		log(LL_VERBOSE, "criterion value: %8.3f", value);
	}

	this->_min_c = min_c;

	return this->_min_c->output();
}

/**
 * Validate a set of predicted labels against the ground truth.
 *
 * @param Y_pred
 */
void ClusteringModel::validate(const std::vector<int>& Y_pred)
{
	// compute purity
	float purity = 0;

	int c = this->_input.labels().size();
	int n = this->_input.entries().size();
	int k = this->_min_c->num_clusters();

	for ( int i = 0; i < k; i++ ) {
		int max_correct = 0;

		for ( int j = 0; j < c; j++ ) {
			int num_correct = 0;

			for ( int p = 0; p < n; p++ ) {
				if ( Y_pred[p] == i && this->_input.entries()[p].label == this->_input.labels()[j] ) {
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
 * @param Y_pred
 */
void ClusteringModel::print_results(const std::vector<int>& Y_pred) const
{
	log(LL_VERBOSE, "Results");

	for ( size_t i = 0; i < this->_input.entries().size(); i++ ) {
		int y_pred = Y_pred[i];
		const DataEntry& entry = this->_input.entries()[i];

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
