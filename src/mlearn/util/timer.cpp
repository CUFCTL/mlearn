/**
 * @file util/timer.cpp
 *
 * Implementation of the timer.
 */
#include <cassert>
#include "mlearn/util/logger.h"
#include "mlearn/util/timer.h"



namespace ML {



std::vector<timer_item_t> Timer::_items;
int Timer::_level = 0;



/**
 * Start a new timer item.
 *
 * @param name
 */
void Timer::push(const std::string& name)
{
	timer_item_t item;
	item.name = name;
	item.level = _level;
	item.start = std::chrono::system_clock::now();
	item.duration = -1;

	_items.push_back(item);
	_level++;

	Logger::log(LogLevel::Verbose, "%*s%s", 2 * item.level, "", item.name.c_str());
}



/**
 * Stop the most recent timer item which is still running.
 *
 * @return duration of the timer item
 */
float Timer::pop()
{
	std::vector<timer_item_t>::reverse_iterator iter;

	for ( iter = _items.rbegin(); iter != _items.rend(); iter++ ) {
		if ( iter->duration == -1 ) {
			break;
		}
	}

	assert(iter != _items.rend());

	iter->end = std::chrono::system_clock::now();
	iter->duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter->end - iter->start).count() / 1000.0f;

	_level--;

	return iter->duration;
}



/**
 * Print all timer items.
 */
void Timer::print()
{
	std::vector<timer_item_t>::iterator iter;

	// determine the maximum string length
	int max_len = 0;

	for ( iter = _items.begin(); iter != _items.end(); iter++ ) {
		int len = 2 * iter->level + iter->name.size();

		if ( max_len < len ) {
			max_len = len;
		}
	}

	// print timer items
	Logger::log(LogLevel::Verbose, "Timing");
	Logger::log(LogLevel::Verbose, "%-*s  %s", max_len, "Name", "Duration (s)");
	Logger::log(LogLevel::Verbose, "%-*s  %s", max_len, "----", "------------");

	for ( iter = _items.begin(); iter != _items.end(); iter++ ) {
		Logger::log(LogLevel::Verbose, "%*s%-*s  % 12.3f",
			2 * iter->level, "",
			max_len - 2 * iter->level, iter->name.c_str(),
			iter->duration);
	}
	Logger::log(LogLevel::Verbose, "");
}



}
