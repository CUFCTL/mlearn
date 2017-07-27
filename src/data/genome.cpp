/**
 * @file data/genome.cpp
 *
 * Implementation of the genome type.
 *
 * The following formats are supported:
 * - RNA-seq data extracted to binary file
 */
#include <cctype>
#include <fstream>
#include "data/genome.h"
#include "util/logger.h"

/**
 * Construct a genome.
 */
Genome::Genome()
{
	this->_gene_count = 0;
	this->_expr_lvls = nullptr;
}

/**
 * Destruct a genome.
 */
Genome::~Genome()
{
	delete this->_expr_lvls;
}

/**
 * Load RNA-seq data from a binary file
 *
 * @param path
 */
void Genome::load_rna_seq(const std::string& path)
{
	std::ifstream file(path, std::ifstream::in);

	std::streampos fsize = file.tellg();
    file.seekg(0, std::ios::end);
    fsize = file.tellg() - fsize;

    this->_gene_count = (int)fsize / sizeof(float);
    this->_expr_lvls = new float[this->_gene_count];

	file.seekg(0);
	file.read(reinterpret_cast<char *>(this->_expr_lvls), this->_gene_count * sizeof(float));
	file.close();
}
