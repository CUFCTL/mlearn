/**
 * @file data/dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include "mlearn/data/dataset.h"
#include "mlearn/util/logger.h"



namespace ML {



/**
 * Construct a dataset from a data iterator.
 *
 * @param iter
 */
Dataset::Dataset(DataIterator *iter)
{
	_iter = iter;

	// construct entries
	_entries = iter->entries();

	// construct classes, labels
	for ( const DataEntry& entry : _entries ) {
		// search for class by name
		size_t j = 0;
		while ( j < _classes.size() && _classes[j] != entry.label ) {
			j++;
		}

		// append class if not found
		if ( j == _classes.size() ) {
			_classes.push_back(entry.label);
		}

		// append label
		_labels.push_back(j);
	}
}



/**
 * Load the data matrix X for a dataset, where each column
 * in X is a sample. Each sample must have the same size.
 */
Matrix Dataset::load_data() const
{
	// construct data matrix
	int m = _iter->sample_size();
	int n = _iter->num_samples();
	Matrix X = Matrix(m, n);

	// map each sample to a column in X
	for ( int i = 0; i < n; i++ ) {
		_iter->sample(X, i);
	}

	return X;
}



/**
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(std::ofstream& file)
{
	// save path
	IODevice::save(file, _path);

	// save classes
	IODevice::save(file, _classes.size());

	for ( auto& name : _classes ) {
		IODevice::save(file, name);
	}

	// save entries
	IODevice::save(file, _entries.size());

	for ( auto& entry : _entries ) {
		IODevice::save(file, entry.label);
		IODevice::save(file, entry.name);
	}

	// save labels
	IODevice::save(file, _labels.size());

	for ( auto& label : _labels ) {
		IODevice::save(file, label);
	}
}



/**
 * Load a dataset from a file.
 *
 * @param file
 */
void Dataset::load(std::ifstream& file)
{
	// read path
	IODevice::load(file, _path);

	// read classes
	int num_classes;
	IODevice::load(file, num_classes);

	_classes.reserve(num_classes);

	for ( int i = 0; i < num_classes; i++ ) {
		std::string name;
		IODevice::load(file, name);

		_classes.push_back(name);
	}

	// read entries
	int num_entries;
	IODevice::load(file, num_entries);

	_entries.reserve(num_entries);

	for ( int i = 0; i < num_entries; i++ ) {
		DataEntry entry;
		IODevice::load(file, entry.label);
		IODevice::load(file, entry.name);

		_entries.push_back(entry);
	}

	// read labels
	int num_labels;
	IODevice::load(file, num_labels);

	_labels.reserve(num_labels);

	for ( int i = 0; i < num_labels; i++ ) {
		int label;
		IODevice::load(file, label);

		_labels.push_back(label);
	}
}



/**
 * Print information about a dataset.
 */
void Dataset::print() const
{
	// print path
	log(LL_VERBOSE, "path: %s", _path.c_str());
	log(LL_VERBOSE, "");

	// print classes
	log(LL_VERBOSE, "%d classes", _classes.size());

	for ( const std::string& name : _classes ) {
		log(LL_VERBOSE, "%s", name.c_str());
	}
	log(LL_VERBOSE, "");

	// print entries
	log(LL_VERBOSE, "%d entries", _entries.size());

	for ( const DataEntry& entry : _entries ) {
		log(LL_VERBOSE, "%-8s  %s", entry.label.c_str(), entry.name.c_str());
	}
	log(LL_VERBOSE, "");
}



}
