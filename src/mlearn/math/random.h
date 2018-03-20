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
	static std::uniform_int_distribution<int> _Ui;
	static std::uniform_real_distribution<float> _Ur;
	static std::normal_distribution<float> _N;

public:
	static void seed(unsigned int value=0);
	static int uniform_int(int a, int b);
	static float uniform_real(float a=0, float b=1);
	static float normal(float mu=0, float sigma=1);
};

}

#endif
