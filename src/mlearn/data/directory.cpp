/**
 * @file data/directory.cpp
 *
 * Implementation of the directory type.
 */
#include <cstring>
#include <dirent.h>
#include "directory.h"



namespace mlearn {



/**
 * Get whether an entry is a file, excluding "." and "..".
 *
 * @param entry
 */
int is_file(const struct dirent *entry)
{
	return (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0);
}



/**
 * Construct a directory.
 *
 * @param path
 */
Directory::Directory(const std::string& path)
{
	// construct entries
	_path = path;

	struct dirent **files;
	int num_entries = scandir(_path.c_str(), &files, is_file, alphasort);

	if ( num_entries < 0 ) {
		perror("scandir");
		exit(1);
	}

	_entries.reserve(num_entries);

	for ( int i = 0; i < num_entries; i++ ) {
		_entries.push_back(files[i]->d_name);
	}

	// cleanup
	for ( int i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);
}



}
