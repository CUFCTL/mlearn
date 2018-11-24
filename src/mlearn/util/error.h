/**
 * @file util/error.h
 */
#ifndef MLEARN_UTIL_ERROR_H
#define MLEARN_UTIL_ERROR_H

#include <stdexcept>



namespace mlearn {



#define CHECK_ERROR(condition, message)  \
	if ( !(condition) )                   \
	{                                     \
		throw std::runtime_error(message); \
	}



}

#endif
