/**
 * @file preprocessing/scaler.cpp
 *
 * Implementation of the scaler type.
 */
#include "mlearn/preprocessing/scaler.h"



namespace ML {



Scaler::Scaler(bool with_mean, bool with_std)
{
	_with_mean = with_mean;
	_with_std = with_std;
}



void Scaler::fit(const Matrix& X)
{
	// compute row-wise mean of X
	if ( _with_mean )
	{
		_mean = X.mean_column();
	}

	// compute row-wise stddev of X
	// TODO
}



void Scaler::transform(Matrix& X)
{
	// subtract row-wise mean from X
	if ( _with_mean )
	{
		X.subtract_columns(_mean);
	}

	// scale X by row-wise stddev
	// TODO
}



void Scaler::save(IODevice& file) const
{
	file << _with_mean;
	file << _with_std;
	file << _mean;
	file << _std;
}



void Scaler::load(IODevice& file)
{
	file >> _with_mean;
	file >> _with_std;
	file >> _mean;
	file >> _std;
}



}
