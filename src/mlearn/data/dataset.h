/**
 * @file data/dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef MLEARN_DATA_DATASET_H
#define MLEARN_DATA_DATASET_H

#include <fstream>
#include <string>
#include <vector>
#include "mlearn/data/dataiterator.h"
#include "mlearn/math/matrix.h"
#include "mlearn/util/iodevice.h"



namespace mlearn {



class Dataset {
public:
	static void train_test_split(
		const Matrix& X, const std::vector<int>& y,
		float test_size,
		Matrix& X_train, std::vector<int>& y_train,
		Matrix& X_test, std::vector<int>& y_test
	);

	Dataset(DataIterator *iter);
	Dataset() {};

	const std::string& path() const { return _path; }
	const std::vector<std::string>& classes() const { return _classes; }
	const std::vector<DataEntry>& entries() const { return _entries; }
	const std::vector<int>& labels() const { return _labels; }

	Matrix load_data() const;
	void print() const;

	friend IODevice& operator<<(IODevice& file, Dataset& dataset);
	friend IODevice& operator>>(IODevice& file, Dataset& dataset);

private:
	DataIterator *_iter;
	std::string _path;
	std::vector<std::string> _classes;
	std::vector<DataEntry> _entries;
	std::vector<int> _labels;
};



}

#endif
