/**
 * @file test_data.cpp
 *
 * Test suite for the data types.
 */
#include <iostream>
#include <memory>
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
	std::unique_ptr<DataIterator> iter;

	if ( args.data_type == "genome" ) {
		iter.reset(new Genome());
	}
	else if ( args.data_type == "image" ) {
		iter.reset(new Image());
	}
	else {
		std::cerr << "error: data type must be 'genome' or 'image'\n";
		exit(1);
	}

	// map a sample to a column vector
	iter->load(args.infile);

	Matrix x(iter->size(), 1);

	iter->to_matrix(x, 0);

	// map the column vector to a sample
	iter->from_matrix(x, 0);
	iter->save(args.outfile);

	return 0;
}
