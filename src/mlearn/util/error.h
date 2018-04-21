/**
 * @file util/error.h
 */
#ifndef ERROR_H
#define ERROR_H

#include <stdexcept>



namespace ML {



#define CHECK_ERROR(condition, message)  \
	if ( !(condition) )                   \
	{                                     \
		throw std::runtime_error(message); \
	}



}

#endif
