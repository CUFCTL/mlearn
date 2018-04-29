/**
 * @file util/error.h
 */
#ifndef ML_UTIL_ERROR_H
#define ML_UTIL_ERROR_H

#include <stdexcept>



namespace ML {



#define CHECK_ERROR(condition, message)  \
	if ( !(condition) )                   \
	{                                     \
		throw std::runtime_error(message); \
	}



}

#endif
