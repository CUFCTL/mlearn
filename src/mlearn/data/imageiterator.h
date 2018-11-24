/**
 * @file data/imageiterator.h
 *
 * Interface definitions for the image iterator.
 */
#ifndef MLEARN_DATA_IMAGEITERATOR_H
#define MLEARN_DATA_IMAGEITERATOR_H

#include <memory>
#include "mlearn/data/dataiterator.h"



namespace mlearn {



class ImageIterator : public DataIterator {
public:
	ImageIterator(const std::string& path);
	~ImageIterator() {};

	int num_samples() const { return _entries.size(); }
	int sample_size() const { return _channels * _width * _height; }
	const std::vector<DataEntry>& entries() const { return _entries; }

	void sample(Matrix& X, int i);

private:
	void load(int i);

	std::string _path;
	std::vector<DataEntry> _entries;

	int _channels;
	int _width;
	int _height;
	int _max_value;
	std::unique_ptr<unsigned char[]> _pixels;
};



}

#endif
