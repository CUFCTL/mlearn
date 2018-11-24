/**
 * @file util/logger.h
 *
 * Interface definitions for the logger.
 */
#ifndef MLEARN_UTIL_LOGGER_H
#define MLEARN_UTIL_LOGGER_H



namespace mlearn {



enum class LogLevel {
	Error   = 0,
	Warn    = 1,
	Info    = 2,
	Verbose = 3,
	Debug   = 4
};



class Logger {
public:
	static LogLevel LEVEL;

	static bool test(LogLevel level) { return (level <= LEVEL); }
	static void log(LogLevel level, const char *format, ...);
};



}

#endif
