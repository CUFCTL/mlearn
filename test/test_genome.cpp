/**
 * @file test_genome.cpp
 *
 * Test suite for the genome data type.
 */
#include <iostream>
#include <mlearn.h>

using namespace ML;

int main(int argc, char **argv)
{
	if ( argc != 3 ) {
		std::cerr << "usage: ./test-genome [infile] [outfile]\n";
		exit(1);
	}

	const char *FILENAME_IN = argv[1];
	const char *FILENAME_OUT = argv[2];

	// map an genome to a column vector
	Genome genome;
	genome.load(FILENAME_IN);

	Matrix x("x", genome.size(), 1);

	genome.to_matrix(x, 0);

	// map a column vector to an genome
	genome.from_matrix(x, 0);
	genome.save(FILENAME_OUT);

	return 0;
}
