/**
 * @file data/dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <string>
#include <vector>
#include "mlearn/data/dataiterator.h"
#include "mlearn/math/matrix.h"

namespace ML {

class Dataset {
private:
	DataIterator *_iter;
	std::string _path;
	std::vector<DataLabel> _labels;
	std::vector<DataEntry> _entries;

public:
	Dataset(DataIterator *iter);
	Dataset() {};

	inline const std::string& path() const { return this->_path; }
	inline const std::vector<DataLabel>& labels() const { return this->_labels; }
	inline const std::vector<DataEntry>& entries() const { return this->_entries; }

	Matrix load_data() const;

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print() const;
};

}

#endif
