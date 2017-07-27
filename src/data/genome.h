/**
 * @file data/genome.h
 *
 * Interface definitions for RNA-seq data (GTEx dataset).
 */
#ifndef GENOME_H
#define GENOME_H

#include "data/datatype.h"

namespace ML {

class Genome : public DataType {
private:
	int _num_genes;
	float *_values;

public:
	Genome();
	~Genome();

	inline int size() const { return this->_num_genes; }
	void to_matrix(Matrix& X, int i) const;
	void from_matrix(Matrix& X, int i);

	void load(const std::string& path);
	void save(const std::string& path);
};

}

#endif
