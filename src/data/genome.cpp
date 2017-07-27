/**
 * @file data/genome.cpp
 *
 * Implementation of the genome type.
 *
 * The following formats are supported:
 * - RNA-seq data extracted to binary file
 */
#include <cassert>
#include <fstream>
#include "data/genome.h"
#include "util/logger.h"

namespace ML {

/**
 * Construct a genome.
 */
Genome::Genome()
{
	this->_num_genes = 0;
	this->_values = nullptr;
}

/**
 * Destruct a genome.
 */
Genome::~Genome()
{
	delete this->_values;
}

/**
 * Load a genome sample into a column of a data matrix.
 *
 * @param X
 * @param i
 */
void Genome::to_matrix(Matrix& X, int i) const
{
	assert(X.rows() == this->size());

	for ( int j = 0; j < X.rows(); j++ ) {
		X.elem(j, i) = this->_values[j];
	}
}

/**
 * Load a genome sample from a column of a data matrix.
 *
 * @param X
 * @param i
 */
void Genome::from_matrix(Matrix& X, int i)
{
	assert(X.rows() == this->size());

	for ( int j = 0; j < X.rows(); j++ ) {
		this->_values[j] = X.elem(j, i);
	}
}

/**
 * Load a genome sample from a binary file.
 *
 * @param path
 */
void Genome::load(const std::string& path)
{
	std::ifstream file(path, std::ifstream::in);

	// determine the size of the genome sample
	std::streampos fsize = file.tellg();
	file.seekg(0, std::ios::end);
	fsize = file.tellg() - fsize;
	file.seekg(0);

	// verify that the genome sizes are equal (if reloading)
	int num = (int)fsize / sizeof(float);

	if ( this->_values == nullptr ) {
		this->_values = new float[num];
	}
	else if ( num != this->size() ) {
		log(LL_ERROR, "error: unequal sizes on genome reload\n");
		exit(1);
	}

	this->_num_genes = num;

	// read genome data
	file.read(reinterpret_cast<char *>(this->_values), this->_num_genes * sizeof(float));

	file.close();
}

/**
 * Save a genome sample to a binary file.
 *
 * @param path
 */
void Genome::save(const std::string& path)
{
	std::ofstream file(path, std::ofstream::out);

	file.write(reinterpret_cast<char *>(this->_values), this->_num_genes);

	file.close();
}

}
