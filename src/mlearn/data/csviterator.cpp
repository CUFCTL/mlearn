/**
 * @file data/csviterator.cpp
 *
 * Implementation of the CSV iterator.
 */
#include <cassert>
#include <fstream>
#include "mlearn/data/csviterator.h"



namespace ML {



/**
 * Construct a CSV iterator from a file.
 *
 * Each line in the file should be an observation
 * with the features followed by the label.
 *
 * @param filename
 */
CSVIterator::CSVIterator(const std::string& filename)
{
	// open file
	std::ifstream file(filename, std::ifstream::in);

	int m;
	int n;
	file >> n >> m;

	_size = m;
	_data.reset(new float[m * n]);

	// construct entries
	_entries.reserve(n);

	for ( int i = 0; i < n; i++ ) {
		// read data
		for ( int j = 0; j < m; j++ ) {
			file >> _data[i * m + j];
		}

		// construct entry
		std::string name = std::to_string(i);

		std::string label;
		file >> label;

		// append entry
		_entries.push_back(DataEntry {
			label,
			name
		});
	}
}



/**
 * Load a sample into a column of a data matrix.
 *
 * @param X
 * @param i
 */
void CSVIterator::sample(Matrix& X, int i)
{
	assert(X.rows() == sample_size());

	for ( int j = 0; j < X.rows(); j++ ) {
		X.elem(j, i) = (float) _data[i * _size + j];
	}
}



}
