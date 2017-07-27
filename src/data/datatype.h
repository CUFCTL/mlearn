/**
 * @file data/datatype.h
 *
 * Definition of the DataType interface.
 */
#ifndef DATATYPE_H
#define DATATYPE_H

#include <string>
#include "math/matrix.h"

namespace ML {

class DataType {
public:
	virtual ~DataType() {};

	virtual int size() const = 0;
	virtual void to_matrix(Matrix& X, int i) const = 0;
	virtual void from_matrix(Matrix& X, int i) = 0;

	virtual void load(const std::string& path) = 0;
	virtual void save(const std::string& path) = 0;
};

}

#endif
