/**
 * @file data/dataiterator.h
 *
 * Definition of the DataIterator interface.
 */
#ifndef DATAITERATOR_H
#define DATAITERATOR_H

#include <string>
#include "mlearn/math/matrix.h"

namespace ML {

class DataIterator {
public:
	virtual ~DataIterator() {};

	virtual int size() const = 0;
	virtual void to_matrix(Matrix& X, int i) const = 0;
	virtual void from_matrix(Matrix& X, int i) = 0;

	virtual void load(const std::string& path) = 0;
	virtual void save(const std::string& path) = 0;
};

}

#endif
