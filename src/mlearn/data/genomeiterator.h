/**
 * @file data/genomeiterator.h
 *
 * Interface definitions for the genome iterator.
 */
#ifndef GENOMEITERATOR_H
#define GENOMEITERATOR_H

#include <memory>
#include "mlearn/data/dataiterator.h"

namespace ML {

class GenomeIterator : public DataIterator {
private:
	std::string _path;
	std::vector<DataEntry> _entries;
	std::vector<DataLabel> _labels;

	int _num_genes;
	std::unique_ptr<float[]> _genes;

	void load(int i);

public:
	GenomeIterator(const std::string& path);
	~GenomeIterator() {};

	int num_samples() const { return _entries.size(); }
	int sample_size() const { return _num_genes; }
	const std::vector<DataEntry>& entries() const { return _entries; }
	const std::vector<DataLabel>& labels() const { return _labels; }

	void sample(Matrix& X, int i);
};

}

#endif
