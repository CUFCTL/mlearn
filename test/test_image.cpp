/**
 * @file test_image.cpp
 *
 * Test suite for the image data type.
 */
#include <iostream>
#include <mlearn.h>

using namespace ML;

int main(int argc, char **argv)
{
	if ( argc != 3 ) {
		std::cerr << "usage: ./test-image [infile] [outfile]\n";
		exit(1);
	}

	const char *FILENAME_IN = argv[1];
	const char *FILENAME_OUT = argv[2];

	// map an image to a column vector
	Image image;
	image.load(FILENAME_IN);

	Matrix x("x", image.size(), 1);

	image.to_matrix(x, 0);

	// map a column vector to an image
	image.from_matrix(x, 0);
	image.save(FILENAME_OUT);

	return 0;
}
