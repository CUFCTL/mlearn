/**
 * @file math/random.cpp
 *
 * Implementation of random number generator.
 */
#include <chrono>
#include "mlearn/math/random.h"



namespace ML {



std::default_random_engine Random::_rng;
std::uniform_int_distribution<int> Random::_Ui;
std::uniform_real_distribution<float> Random::_Ur;
std::normal_distribution<float> Random::_N;



/**
 * Seed the global random number engine with the current time.
 */
void Random::seed(unsigned int value)
{
    if ( value == 0 ) {
        value = std::chrono::system_clock::now().time_since_epoch().count();
    }

    _rng.seed(value);
}



/**
 * Generate an integer from a uniform (a, b) distribution.
 *
 * @param a
 * @param b
 */
int Random::uniform_int(int a, int b)
{
    return _Ui(_rng) % (b - a) + a;
}



/**
 * Generate a real number from a uniform (a, b) distribution.
 *
 * @param a
 * @param b
 */
float Random::uniform_real(float a, float b)
{
    return _Ur(_rng) * (b - a) + a;
}



/**
 * Generate a real number from a normal (mu, sigma^2) distribution.
 *
 * @param mu
 * @param sigma
 */
float Random::normal(float mu, float sigma)
{
    return mu + _N(_rng) * sigma;
}



}
