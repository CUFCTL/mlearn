/**
 * @file math/random.h
 *
 * Interface definitions for the random number generator.
 */
#ifndef RANDOM_H
#define RANDOM_H

#include <random>

namespace ML {

class Random {
private:
	static std::default_random_engine _rng;
	static std::uniform_int_distribution<int> _U;
	static std::normal_distribution<float> _N;

public:
	static void seed(unsigned int value=0);
	static int uniform_int(int a, int b);
	static float normal(float mu=0, float sigma=1);
};

}

#endif
