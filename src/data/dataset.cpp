/**
 * @file data/dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include <dirent.h>
#include <fstream>
#include <memory>
#include "data/dataset.h"
#include "util/logger.h"

namespace ML {

/**
 * Get whether an entry is a file, excluding "." and "..".
 *
 * @param entry
 */
int is_file(const struct dirent *entry)
{
	std::string name(entry->d_name);

	return (name != "." && name != "..");
}

/**
 * Read an integer from a binary file.
 *
 * @param file
 */
int read_int(std::ifstream& file)
{
	int n;
	file.read(reinterpret_cast<char *>(&n), sizeof(int));

	return n;
}

/**
 * Read a string from a binary file.
 *
 * @param file
 */
std::string read_string(std::ifstream& file)
{
	int num = read_int(file);

	std::unique_ptr<char[]> buffer(new char[num]);
	file.read(buffer.get(), num);

	std::string str(buffer.get());

	return str;
}

/**
 * Write an integer to a binary file.
 *
 * @param file
 */
void write_int(int n, std::ofstream& file)
{
	file.write(reinterpret_cast<char *>(&n), sizeof(int));
}

/**
 * Write a string to a file.
 *
 * @param str
 * @param file
 */
void write_string(const std::string& str, std::ofstream& file)
{
	int num = str.size() + 1;

	write_int(num, file);
	file.write(str.c_str(), num);
}

/**
 * Construct a dataset from a file or a directory.
 *
 * If path is a directory, each file in the
 * directory is treated as an observation. If
 * the data are labeled, the filename for each
 * observation should be formatted as follows:
 *
 * "<class>_<...>"
 *
 * This format is used to determine the label of each
 * file without separate label data, and to group the
 * entries by label.
 *
 * If the path is a file, each line should be an observation
 * with the features followed by the label.
 *
 * @param iter
 * @param path
 * @param is_labeled
 */
Dataset::Dataset(DataIterator *iter, const std::string& path, bool is_labeled)
{
	this->_iter = iter;
	this->_path = path;

	if ( iter != nullptr ) {
		// get list of files
		struct dirent **files;
		int num_entries = scandir(this->_path.c_str(), &files, is_file, alphasort);

		if ( num_entries <= 0 ) {
			perror("scandir");
			exit(1);
		}

		// construct entries
		for ( int i = 0; i < num_entries; i++ ) {
			// construct entry name
			std::string name(files[i]->d_name);

			// construct label name
			DataLabel label = is_labeled
				? name.substr(0, name.find_first_of('_'))
				: "";

			// append entry
			this->_entries.push_back(DataEntry {
				label,
				name
			});
		}

		// construct labels
		if ( is_labeled ) {
			for ( const DataEntry& entry : this->_entries ) {
				// search labels for label name
				size_t j = 0;
				while ( j < this->_labels.size() && this->_labels[j] != entry.label ) {
					j++;
				}

				// append label if not found
				if ( j == this->_labels.size() ) {
					this->_labels.push_back(entry.label);
				}
			}
		}

		// clean up
		for ( int i = 0; i < num_entries; i++ ) {
			free(files[i]);
		}
		free(files);
	}
	else {
		std::ifstream file(this->_path, std::ifstream::in);

		int m;
		int n;
		file >> n >> m;

		// construct entries
		for ( int i = 0; i < n; i++ ) {
			// skip data
			float data;
			for ( int j = 0; j < m; j++ ) {
				file >> data;
			}

			// construct entry
			std::string name = std::to_string(i);
			DataLabel label;

			if ( is_labeled ) {
				file >> label;
			}

			// append entry
			this->_entries.push_back(DataEntry {
				label,
				name
			});
		}

		// construct labels
		if ( is_labeled ) {
			for ( const DataEntry& entry : this->_entries ) {
				// search labels for label name
				size_t j = 0;
				while ( j < this->_labels.size() && this->_labels[j] != entry.label ) {
					j++;
				}

				// append label if not found
				if ( j == this->_labels.size() ) {
					this->_labels.push_back(entry.label);
				}
			}
		}
	}
}

/**
 * Load the data matrix X for a dataset. Each column
 * in X is an observation. Every observation in X must
 * have the same dimensionality.
 */
Matrix Dataset::load_data() const
{
	Matrix X;

	if ( this->_iter != nullptr ) {
		// get the size of the first sample
		this->_iter->load(this->_path + "/" + this->_entries[0].name);

		// construct data matrix
		int m = this->_iter->size();
		int n = this->_entries.size();
		X = Matrix(m, n);

		// map each sample to a column in X
		this->_iter->to_matrix(X, 0);

		for ( int i = 1; i < n; i++ ) {
			this->_iter->load(this->_path + "/" + this->_entries[i].name);
			this->_iter->to_matrix(X, i);
		}
	}
	else {
		std::ifstream file(this->_path, std::ifstream::in);

		// construct data matrix
		int m;
		int n;
		file >> n >> m;
		X = Matrix(m, n);

		// map each sample (line) to a column in X
		for ( int i = 0; i < n; i++ ) {
			for ( int j = 0; j < m; j++ ) {
				file >> X.elem(j, i);
			}

			// skip label
			DataLabel label;
			file >> label;
		}
	}

	return X;
}

/**
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(std::ofstream& file)
{
	// save path
	write_string(this->_path.c_str(), file);

	// save labels
	int num_labels = this->_labels.size();
	write_int(num_labels, file);

	for ( const DataLabel& label : this->_labels ) {
		write_string(label.c_str(), file);
	}

	// save entries
	int num_entries = this->_entries.size();
	write_int(num_entries, file);

	for ( const DataEntry& entry : this->_entries ) {
		write_string(entry.label.c_str(), file);
		write_string(entry.name.c_str(), file);
	}
}

/**
 * Load a dataset from a file.
 *
 * @param file
 */
void Dataset::load(std::ifstream& file)
{
	// read path
	this->_path = read_string(file);

	// read labels
	int num_labels = read_int(file);

	for ( int i = 0; i < num_labels; i++ ) {
		DataLabel label(read_string(file));

		this->_labels.push_back(label);
	}

	// read entries
	int num_entries = read_int(file);

	for ( int i = 0; i < num_entries; i++ ) {
		DataEntry entry;
		entry.label = read_string(file);
		entry.name = read_string(file);

		this->_entries.push_back(entry);
	}
}

/**
 * Print information about a dataset.
 */
void Dataset::print() const
{
	// print path
	log(LL_VERBOSE, "path: %s", this->_path.c_str());
	log(LL_VERBOSE, "");

	// print labels
	log(LL_VERBOSE, "%d classes", this->_labels.size());

	for ( const DataLabel& label : this->_labels ) {
		log(LL_VERBOSE, "%s", label.c_str());
	}
	log(LL_VERBOSE, "");

	// print entries
	log(LL_VERBOSE, "%d entries", this->_entries.size());

	for ( const DataEntry& entry : this->_entries ) {
		log(LL_VERBOSE, "%-8s  %s", entry.label.c_str(), entry.name.c_str());
	}
	log(LL_VERBOSE, "");
}

}
