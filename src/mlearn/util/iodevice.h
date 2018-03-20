/**
 * @file util/iodevice.h
 *
 * Interface definitions for the I/O device type.
 *
 * This class provides methods for loading and saving
 * several common data types to a file, as well as printing
 * to a text stream.
 */
#ifndef IODEVICE_H
#define IODEVICE_H

#include <fstream>
#include <iostream>



namespace ML {



class IODevice {
public:
	virtual ~IODevice() {};

	virtual void save(std::ofstream& file) {};
	virtual void load(std::ifstream& file) {};
	virtual void print() const {};
};



}

#endif
