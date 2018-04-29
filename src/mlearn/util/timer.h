/**
 * @file util/timer.h
 *
 * Interface definitions for the timer.
 */
#ifndef ML_UTIL_TIMER_H
#define ML_UTIL_TIMER_H

#include <chrono>
#include <string>
#include <vector>



namespace ML {



typedef struct {
	std::string name;
	int level;
	std::chrono::system_clock::system_clock::time_point start;
	std::chrono::system_clock::system_clock::time_point end;
	float duration;
} timer_item_t;



class Timer {
public:
	static void push(const std::string& name);
	static float pop();
	static void print();

private:
	static std::vector<timer_item_t> _items;
	static int _level;
};



}

#endif
