/**
 * @file util/iodevice.cpp
 *
 * Implementation of I/O device type.
 */
#include <memory>
#include "mlearn/util/iodevice.h"



namespace ML {



void IODevice::save(std::ofstream& file, int val)
{
	file.write(reinterpret_cast<char *>(&val), sizeof(int));
}



void IODevice::save(std::ofstream& file, const std::string& val)
{
	int len = val.size() + 1;

	file << len;
	file.write(val.c_str(), len);
}



void IODevice::load(std::ifstream& file, int& val)
{
	file.read(reinterpret_cast<char *>(&val), sizeof(int));
}



void IODevice::load(std::ifstream& file, std::string& val)
{
	int len;
	file >> len;

	std::unique_ptr<char[]> buffer(new char[len]);
	file.read(buffer.get(), len);

	val = std::string(buffer.get());
}



}
