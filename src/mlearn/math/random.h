/**
 * @file math/random.h
 *
 * Interface definitions for the random number generator.
 */
#ifndef MLEARN_MATH_RANDOM_H
#define MLEARN_MATH_RANDOM_H

#include <random>
#include <vector>



namespace mlearn {



class Random {
public:
	static void seed(unsigned int value=0);
	static int uniform_int(int a, int b);
	static float uniform_real(float a=0, float b=1);
	static float normal(float mu=0, float sigma=1);

	template<class T>
	static void shuffle(std::vector<T>& v);
private:
	static std::default_random_engine _rng;
	static std::uniform_int_distribution<int> _Ui;
	static std::uniform_real_distribution<float> _Ur;
	static std::normal_distribution<float> _N;
};



template<class T>
void Random::shuffle(std::vector<T>& v)
{
	for ( int i = 0; i < v.size(); i++ )
	{
		int j = Random::uniform_int(i, v.size());

		if ( i != j )
		{
			std::swap(v[i], v[j]);
		}
	}
}



}

#endif
