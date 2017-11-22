/**
 * @file data/imageiterator.h
 *
 * Interface definitions for the image iterator.
 */
#ifndef IMAGEITERATOR_H
#define IMAGEITERATOR_H

#include <memory>
#include "mlearn/data/dataiterator.h"

namespace ML {

class ImageIterator : public DataIterator {
private:
	std::string _path;
	std::vector<DataEntry> _entries;

	int _channels;
	int _width;
	int _height;
	int _max_value;
	std::unique_ptr<unsigned char[]> _pixels;

	void load(int i);

public:
	ImageIterator(const std::string& path);
	~ImageIterator() {};

	int num_samples() const { return _entries.size(); }
	int sample_size() const { return _channels * _width * _height; }
	const std::vector<DataEntry>& entries() const { return _entries; }

	void sample(Matrix& X, int i);
};

}

#endif
