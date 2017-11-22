/**
 * @file data/dataiterator.h
 *
 * Definition of the DataIterator interface.
 */
#ifndef DATAITERATOR_H
#define DATAITERATOR_H

#include <string>
#include <vector>
#include "mlearn/math/matrix.h"

namespace ML {

typedef std::string DataLabel;

typedef struct {
	DataLabel label;
	std::string name;
} DataEntry;

class DataIterator {
public:
	virtual ~DataIterator() {};

	virtual int num_samples() const = 0;
	virtual int sample_size() const = 0;
	virtual const std::vector<DataEntry>& entries() const = 0;

	virtual void sample(Matrix& X, int i) = 0;
};

}

#endif
