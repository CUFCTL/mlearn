/**
 * @file data/genomeiterator.cpp
 *
 * Implementation of the genome iterator.
 *
 * The following formats are supported:
 * - RNA-seq data extracted to binary file
 */
#include <cassert>
#include <fstream>
#include "mlearn/data/directory.h"
#include "mlearn/data/genomeiterator.h"
#include "mlearn/util/logger.h"

namespace ML {

/**
 * Construct a genome iterator from a directory.
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
GenomeIterator::GenomeIterator(const std::string& path)
{
	// get list of files
	this->_path = path;

	Directory dir(path);

	// construct entries
	this->_entries.reserve(dir.entries().size());

	for ( int i = 0; i < dir.entries().size(); i++ ) {
		// construct entry name
		std::string name(dir.entries()[i]);

		// construct label name
		DataLabel label = name.substr(0, name.find_first_of('_'));

		// append entry
		this->_entries.push_back(DataEntry {
			label,
			name
		});
	}

	this->_num_genes = 0;
	this->_genes.reset();

	// load first sample to get size
	this->load(0);
}

/**
 * Load a sample into a column of a data matrix.
 *
 * @param X
 * @param i
 */
void GenomeIterator::sample(Matrix& X, int i)
{
	assert(X.rows() == this->sample_size());

	this->load(i);

	for ( int j = 0; j < X.rows(); j++ ) {
		X.elem(j, i) = (float) this->_genes[j];
	}
}

/**
 * Load a genome sample from a binary file.
 *
 * @param i
 */
void GenomeIterator::load(int i)
{
	// open file
	std::string path = this->_path + "/" + this->_entries[i].name;
	std::ifstream file(path, std::ifstream::in);

	// determine the size of the genome sample
	std::streampos fsize = file.tellg();
	file.seekg(0, std::ios::end);
	fsize = file.tellg() - fsize;
	file.seekg(0);

	// verify that the genome sizes are equal (if reloading)
	int num = (int)fsize / sizeof(float);

	if ( this->_genes == nullptr ) {
		this->_genes.reset(new float[num]);
	}
	else if ( num != this->sample_size() ) {
		log(LL_ERROR, "error: genome \'%s\' has unequal size\n", path.c_str());
		exit(1);
	}

	this->_num_genes = num;

	// read genome data
	file.read(reinterpret_cast<char *>(this->_genes.get()), this->_num_genes * sizeof(float));

	file.close();
}

}
