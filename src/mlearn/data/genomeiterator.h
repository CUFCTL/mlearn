/**
 * @file data/genomeiterator.h
 *
 * Interface definitions for the genome iterator.
 */
#ifndef MLEARN_DATA_GENOMEITERATOR_H
#define MLEARN_DATA_GENOMEITERATOR_H

#include <memory>
#include "mlearn/data/dataiterator.h"



namespace mlearn {



class GenomeIterator : public DataIterator {
public:
	GenomeIterator(const std::string& path);
	~GenomeIterator() {};

	int num_samples() const { return _entries.size(); }
	int sample_size() const { return _num_genes; }
	const std::vector<DataEntry>& entries() const { return _entries; }

	void sample(Matrix& X, int i);

private:
	void load(int i);

	std::string _path;
	std::vector<DataEntry> _entries;

	int _num_genes;
	std::unique_ptr<float[]> _genes;
};



}

#endif
