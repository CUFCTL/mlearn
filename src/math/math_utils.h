/**
 * @file math/math_utils.h
 *
 * Library of helpful math functions.
 */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

namespace ML {

void RNG_seed(unsigned int value=0);
int RNG_int(int a, int b);
float RNG_normal(float mu=0, float sigma=1);

float pow2f(float x);
float pow3f(float x);
float sechf(float x);

}

#endif
