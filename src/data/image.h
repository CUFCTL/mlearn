/**
 * @file data/image.h
 *
 * Interface definitions for the image type.
 */
#ifndef IMAGE_H
#define IMAGE_H

#include <memory>
#include "data/dataiterator.h"

namespace ML {

class Image : public DataIterator {
private:
	int _channels;
	int _width;
	int _height;
	int _max_value;
	std::unique_ptr<unsigned char[]> _pixels;

public:
	Image();
	~Image() {};

	int size() const { return _channels * _width * _height; }
	void to_matrix(Matrix& X, int i) const;
	void from_matrix(Matrix& X, int i);

	void load(const std::string& path);
	void save(const std::string& path);
};

}

#endif
