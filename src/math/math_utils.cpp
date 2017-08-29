/**
 * @file math/math_utils.cpp
 *
 * Library of helpful math functions.
 */
#include <cmath>
#include "math/math_utils.h"

namespace ML {

/**
 * Compute the second power (square) of a number.
 *
 * @param x
 * @return x^2
 */
float pow2(float x)
{
    return powf(x, 2);
}

/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 * @return x^3
 */
float pow3(float x)
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
