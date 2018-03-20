/**
 * @file data/dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include <fstream>
#include <memory>
#include "mlearn/data/dataset.h"
#include "mlearn/util/logger.h"



namespace ML {



/**
 * Read an integer from a binary file.
 *
 * @param file
 */
int read_int(std::ifstream& file)
{
	int n;
	file.read(reinterpret_cast<char *>(&n), sizeof(int));

	return n;
}



/**
 * Read a string from a binary file.
 *
 * @param file
 */
std::string read_string(std::ifstream& file)
{
	int num = read_int(file);

	std::unique_ptr<char[]> buffer(new char[num]);
	file.read(buffer.get(), num);

	std::string str(buffer.get());

	return str;
}



/**
 * Write an integer to a binary file.
 *
 * @param file
 */
void write_int(int n, std::ofstream& file)
{
	file.write(reinterpret_cast<char *>(&n), sizeof(int));
}



/**
 * Write a string to a file.
 *
 * @param str
 * @param file
 */
void write_string(const std::string& str, std::ofstream& file)
{
	int num = str.size() + 1;

	write_int(num, file);
	file.write(str.c_str(), num);
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
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(std::ofstream& file)
{
	// save path
	write_string(_path.c_str(), file);

	// save labels
	int num_classes = _classes.size();
	write_int(num_classes, file);

	for ( const std::string& label : _classes ) {
		write_string(label.c_str(), file);
	}

	// save entries
	int num_entries = _entries.size();
	write_int(num_entries, file);

	for ( const DataEntry& entry : _entries ) {
		write_string(entry.label.c_str(), file);
		write_string(entry.name.c_str(), file);
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
	_path = read_string(file);

	// read labels
	int num_classes = read_int(file);

	_classes.reserve(num_classes);

	for ( int i = 0; i < num_classes; i++ ) {
		std::string label(read_string(file));

		_classes.push_back(label);
	}

	// read entries
	int num_entries = read_int(file);

	_entries.reserve(num_entries);

	for ( int i = 0; i < num_entries; i++ ) {
		DataEntry entry;
		entry.label = read_string(file);
		entry.name = read_string(file);

		_entries.push_back(entry);
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

	for ( const std::string& label : _classes ) {
		log(LL_VERBOSE, "%s", label.c_str());
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
