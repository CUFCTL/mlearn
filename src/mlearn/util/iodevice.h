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
#include <vector>



namespace ML {



class IODevice : public std::fstream {
public:
	IODevice(const std::string& filename, std::ios_base::openmode mode)
		: std::fstream(filename, mode) {};

	IODevice& operator<<(bool val);
	IODevice& operator<<(float val);
	IODevice& operator<<(int val);
	IODevice& operator<<(const std::string& val);
	IODevice& operator>>(bool& val);
	IODevice& operator>>(float& val);
	IODevice& operator>>(int& val);
	IODevice& operator>>(std::string& val);

	template<class T> IODevice& operator<<(const std::vector<T>& v);
	template<class T> IODevice& operator>>(std::vector<T>& v);
};



template<class T>
IODevice& IODevice::operator<<(const std::vector<T>& v)
{
	int size = v.size();
	(*this) << size;

	for ( const T& e : v ) {
		(*this) << e;
	}

	return (*this);
}



template<class T>
IODevice& IODevice::operator>>(std::vector<T>& v)
{
	int size;
	(*this) >> size;

	v.reserve(size);

	for ( int i = 0; i < size; i++ ) {
		T e;
		(*this) >> e;

		v.push_back(e);
	}

	return (*this);
}



}

#endif
