/**
 * @file util/logger.cpp
 *
 * Implementation of the logger.
 */
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include "mlearn/util/logger.h"

namespace ML {



LogLevel Logger::LEVEL { LogLevel::Info };



/**
 * Log a message with a given loglevel.
 *
 * This function uses the same argument format
 * as printf().
 *
 * @param level
 * @param format
 */
void Logger::log(LogLevel level, const char *format, ...)
{
	if ( test(level) ) {
		FILE *stream = (level <= LogLevel::Error)
			? stderr
			: stdout;
		va_list ap;

		time_t t = time(nullptr);
		struct tm *tm = localtime(&t);

		fprintf(stream, "[%04d-%02d-%02d %02d:%02d:%02d] ",
			1900 + tm->tm_year, 1 + tm->tm_mon, tm->tm_mday,
			tm->tm_hour, tm->tm_min, tm->tm_sec);

		va_start(ap, format);
		vfprintf(stream, format, ap);
		va_end(ap);

		printf("\n");
	}
}



}
