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
 * @param feature
 * @param clustering
 */
ClusteringModel::ClusteringModel(FeatureLayer *feature, ClusteringLayer *clustering)
{
	// initialize layers
	this->_feature = feature;
	this->_clustering = clustering;

	// initialize stats
	this->_stats.error_rate = 0.0f;
	this->_stats.extract_time = 0.0f;
	this->_stats.predict_time = 0.0f;

	// log hyperparameters
	log(LL_VERBOSE, "Hyperparameters");

	this->_feature->print();
	this->_clustering->print();

	log(LL_VERBOSE, "");
}

/**
 * Extract features from input data.
 *
 * @param input
 */
void ClusteringModel::extract(const Dataset& input)
{
	timer_push("Feature extraction");

	this->_input = input;

	log(LL_VERBOSE, "Input data: %d samples, %d classes",
		input.entries().size(),
		input.labels().size());

	// get data matrix X
	Matrix X = input.load_data();

	// subtract mean from X
	Matrix mu = X.mean_column();

	X.subtract_columns(mu);

	// project X into feature space
	this->_feature->compute(X, this->_input.entries(), this->_input.labels().size());
	this->_P = this->_feature->project(X);

	// record extraction time
	this->_stats.extract_time = timer_pop();

	log(LL_VERBOSE, "");
}

/**
 * Perform clustering on the input data.
 */
std::vector<int> ClusteringModel::predict()
{
	timer_push("Clustering");

	// run clustering layer
	this->_clustering->compute(this->_P);

	// record prediction time
	this->_stats.predict_time = timer_pop();

	return this->_clustering->output();
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
	int k = this->_clustering->num_clusters();

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
		<< std::setw(12) << std::setprecision(3) << this->_stats.extract_time
		<< std::setw(12) << std::setprecision(3) << this->_stats.predict_time
		<< "\n";
}

}
