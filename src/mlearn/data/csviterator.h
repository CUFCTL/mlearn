/**
 * @file data/csviterator.h
 *
 * Interface definitions for the CSV iterator.
 */
#ifndef CSVITERATOR_H
#define CSVITERATOR_H

#include <memory>
#include "mlearn/data/dataiterator.h"



namespace ML {



class CSVIterator : public DataIterator {
public:
	CSVIterator(const std::string& filename);
	~CSVIterator() {};

	int num_samples() const { return _entries.size(); }
	int sample_size() const { return _size; }
	const std::vector<DataEntry>& entries() const { return _entries; }

	void sample(Matrix& X, int i);

private:
	std::vector<DataEntry> _entries;

	int _size;
	std::unique_ptr<float[]> _data;
};



}

#endif
