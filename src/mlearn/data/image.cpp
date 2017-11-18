/**
 * @file data/image.cpp
 *
 * Implementation of the image type.
 *
 * The following formats are supported:
 * - binary PGM (P5)
 * - binary PPM (P6)
 */
#include <cassert>
#include <cctype>
#include <fstream>
#include "mlearn/data/image.h"
#include "mlearn/util/logger.h"

namespace ML {

/**
 * Construct an image.
 */
Image::Image()
{
	this->_channels = 0;
	this->_width = 0;
	this->_height = 0;
	this->_max_value = 0;
	this->_pixels.reset();
}

/**
 * Load an image into a column of a data matrix.
 *
 * @param X
 * @param i
 */
void Image::to_matrix(Matrix& X, int i) const
{
	assert(X.rows() == this->size());

	for ( int j = 0; j < X.rows(); j++ ) {
		X.elem(j, i) = (float) this->_pixels[j];
	}
}

/**
 * Load an image from a column of a data matrix.
 *
 * @param X
 * @param i
 */
void Image::from_matrix(Matrix& X, int i)
{
	assert(X.rows() == this->size());

	for ( int j = 0; j < X.rows(); j++ ) {
		this->_pixels[j] = (unsigned char) X.elem(j, i);
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
 * @param path
 */
void Image::load(const std::string& path)
{
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
		log(LL_ERROR, "error: cannot read image \'%s\'\n", path.c_str());
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

	if ( this->_pixels == nullptr ) {
		this->_pixels.reset(new unsigned char[num]);
	}
	else if ( num != this->size() ) {
		log(LL_ERROR, "error: unequal sizes on image reload\n");
		exit(1);
	}

	this->_channels = channels;
	this->_width = width;
	this->_height = height;
	this->_max_value = max_value;

	// read pixel data
	file.read(reinterpret_cast<char *>(this->_pixels.get()), num);

	file.close();
}

/**
 * Save an image to a PGM/PPM file.
 *
 * @param path
 */
void Image::save(const std::string& path)
{
	std::ofstream file(path, std::ofstream::out);

	// determine image header
	std::string header;

	if ( this->_channels == 1 ) {
		header = "P5";
	}
	else if ( this->_channels == 3 ) {
		header = "P6";
	}
	else {
		log(LL_ERROR, "error: cannot write image \'%s\'\n", path.c_str());
		exit(1);
	}

	// write image metadata
	file << header << "\n"
		<< this->_width << " "
		<< this->_height << " "
		<< this->_max_value << "\n";

	// write pixel data
	int num = this->_channels * this->_width * this->_height;

	file.write(reinterpret_cast<char *>(this->_pixels.get()), num);

	file.close();
}

}
