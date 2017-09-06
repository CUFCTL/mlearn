/**
 * @file test_data.cpp
 *
 * Test suite for the data types.
 */
#include <iostream>
#include <mlearn.h>

using namespace ML;

typedef struct {
	std::string data_type;
	std::string infile;
	std::string outfile;
} args_t;

int main(int argc, char **argv)
{
	if ( argc != 4 ) {
		std::cerr << "usage: ./test-data [type] [infile] [outfile]\n";
		exit(1);
	}

	args_t args = {
		argv[1],
		argv[2],
		argv[3]
	};

	// initialize data type
	DataType *sample;

	if ( args.data_type == "genome" ) {
		sample = new Genome();
	}
	else if ( args.data_type == "image" ) {
		sample = new Image();
	}

	// map the sample to a column vector
	sample->load(args.infile);

	Matrix x("x", sample->size(), 1);

	sample->to_matrix(x, 0);

	// map the column vector to a sample
	sample->from_matrix(x, 0);
	sample->save(args.outfile);

	// cleanup
	delete sample;

	return 0;
}
