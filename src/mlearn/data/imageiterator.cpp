/**
 * @file data/imageiterator.cpp
 *
 * Implementation of the image iterator.
 *
 * The following image formats are supported:
 * - binary PGM (P5)
 * - binary PPM (P6)
 */
#include <cassert>
#include <cctype>
#include <fstream>
#include "mlearn/data/directory.h"
#include "mlearn/data/imageiterator.h"
#include "mlearn/util/logger.h"



namespace ML {



/**
 * Construct an image iterator from a directory.
 *
 * Each file in the directory is treated as a
 * sample. The filename for each sample should be
 * formatted as follows:
 *
 * "<label>_<...>"
 *
 * This format is used to determine the label of each
 * file without separate label data, and to group the
 * entries by label.
 *
 * @param path
 */
ImageIterator::ImageIterator(const std::string& path)
{
	// get list of files
	_path = path;

	Directory dir(path);

	// construct entries
	_entries.reserve(dir.entries().size());

	for ( auto& name : dir.entries() ) {
		// construct label name
		std::string label = name.substr(0, name.find_first_of('_'));

		// append entry
		_entries.push_back(DataEntry {
			label,
			name
		});
	}

	_channels = 0;
	_width = 0;
	_height = 0;
	_max_value = 0;
	_pixels.reset();

	// load first sample to get size
	load(0);
}



/**
 * Load a sample into a column of a data matrix.
 *
 * @param X
 * @param i
 */
void ImageIterator::sample(Matrix& X, int i)
{
	assert(X.rows() == sample_size());

	load(i);

	for ( int j = 0; j < X.rows(); j++ ) {
		X.elem(j, i) = (float) _pixels[j];
	}
}



/**
 * Helper function to skip comments in a PGM/PPM image.
 *
 * @param file
 */
void skip_to_next_value(std::ifstream& file)
{
	char c = file.get();
	while ( c == '#' || isspace(c) ) {
		if ( c == '#' ) {
			while ( c != '\n' ) {
				c = file.get();
			}
		}
		else {
			while ( isspace(c) ) {
				c = file.get();
			}
		}
	}

	file.unget();
}



/**
 * Load an image from a PGM/PPM file.
 *
 * @param i
 */
void ImageIterator::load(int i)
{
	// open file
	std::string path = _path + "/" + _entries[i].name;
	std::ifstream file(path, std::ifstream::in);

	// read image header
	std::string header;
	file >> header;

	// determine image channels
	int channels;

	if ( header == "P5" ) {
		channels = 1;
	}
	else if ( header == "P6" ) {
		channels = 3;
	}
	else {
		Logger::log(LogLevel::Error, "error: cannot read image \'%s\'\n", path.c_str());
		exit(1);
	}

	skip_to_next_value(file);

	// read image metadata
	int width;
	int height;
	int max_value;

	file >> width;
	skip_to_next_value(file);

	file >> height;
	skip_to_next_value(file);

	file >> max_value;
	file.get();

	// verify that image sizes are equal (if reloading)
	int num = channels * width * height;

	if ( _pixels == nullptr ) {
		_pixels.reset(new unsigned char[num]);
	}
	else if ( num != sample_size() ) {
		Logger::log(LogLevel::Error, "error: image \'%s\' has unequal size\n", path.c_str());
		exit(1);
	}

	_channels = channels;
	_width = width;
	_height = height;
	_max_value = max_value;

	// read pixel data
	file.read(reinterpret_cast<char *>(_pixels.get()), num);

	file.close();
}



}
