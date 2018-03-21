/**
 * @file data/dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include "mlearn/data/dataset.h"
#include "mlearn/util/logger.h"



namespace ML {



IODevice& operator<<(IODevice& file, const DataEntry& entry)
{
	file << entry.label;
	file << entry.name;
	return file;
}



IODevice& operator>>(IODevice& file, DataEntry& entry)
{
	file >> entry.label;
	file >> entry.name;
	return file;
}



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
 * Print information about a dataset.
 */
void Dataset::print() const
{
	// print path
	Logger::log(LogLevel::Verbose, "path: %s", _path.c_str());
	Logger::log(LogLevel::Verbose, "");

	// print classes
	Logger::log(LogLevel::Verbose, "%d classes", _classes.size());

	for ( const std::string& name : _classes ) {
		Logger::log(LogLevel::Verbose, "%s", name.c_str());
	}
	Logger::log(LogLevel::Verbose, "");

	// print entries
	Logger::log(LogLevel::Verbose, "%d entries", _entries.size());

	for ( const DataEntry& entry : _entries ) {
		Logger::log(LogLevel::Verbose, "%-8s  %s", entry.label.c_str(), entry.name.c_str());
	}
	Logger::log(LogLevel::Verbose, "");
}



/**
 * Save a dataset to a file.
 */
IODevice& operator<<(IODevice& file, Dataset& dataset)
{
	file << dataset._path;
	file << dataset._classes;
	file << dataset._entries;
	file << dataset._labels;
	return file;
}



/**
 * Load a dataset from a file.
 */
IODevice& operator>>(IODevice& file, Dataset& dataset)
{
	file >> dataset._path;
	file >> dataset._classes;
	file >> dataset._entries;
	file >> dataset._labels;
	return file;
}



}
