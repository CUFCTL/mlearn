/**
 * @file math/math_utils.cpp
 *
 * Library of helpful math functions.
 */
#include <chrono>
#include <cmath>
#include <random>
#include "math/math_utils.h"

namespace ML {

std::default_random_engine RNG;

/**
 * Seed the global random number engine with the current time.
 */
void RNG_seed(unsigned int value)
{
    if ( value == 0 ) {
        value = std::chrono::system_clock::now().time_since_epoch().count();
    }

    RNG.seed(value);
}

/**
 * Generate an integer from a uniform (a, b) distribution.
 *
 * @param a
 * @param b
 */
int RNG_int(int a, int b)
{
    static std::uniform_int_distribution<int> U;

    return U(RNG) % (a - b) + a;
}

/**
 * Generate a real number from a normal (mu, sigma^2) distribution.
 *
 * @param mu
 * @param sigma
 */
float RNG_normal(float mu, float sigma)
{
    static std::normal_distribution<float> N;

    return mu + N(RNG) * sigma;
}

/**
 * Compute the second power (square) of a number.
 *
 * @param x
 * @return x^2
 */
float pow2f(float x)
{
    return powf(x, 2);
}

/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 * @return x^3
 */
float pow3f(float x)
{
    return powf(x, 3);
}

/**
 * Compute the hyperbolic secant of a number.
 *
 * @param x
 * @return sech(x)
 */
float sechf(float x)
{
    return 1.0f / coshf(x);
}

}
