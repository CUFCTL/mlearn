/**
 * @file preprocessing/scaler.cpp
 *
 * Implementation of the scaler type.
 */
#include "mlearn/preprocessing/scaler.h"



namespace mlearn {



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
	if ( _with_std )
	{
		_std = Matrix::zeros(X.rows(), 1);

		for ( int i = 0; i < X.rows(); i++ )
		{
			for ( int j = 0; j < X.cols(); j++ )
			{
				float temp = X.elem(i, j) - _mean.elem(i);
				_std.elem(i) += temp * temp;
			}
		}

		_std /= X.cols();
		_std.elem_apply(sqrtf);
	}
}



Matrix Scaler::transform(const Matrix& X_) const
{
	Matrix X(X_);

	// subtract row-wise mean from X
	if ( _with_mean )
	{
		X.subtract_columns(_mean);
	}

	// scale X by row-wise stddev
	if ( _with_std )
	{
		for ( int i = 0; i < X.rows(); i++ )
		{
			for ( int j = 0; j < X.cols(); j++ )
			{
				X.elem(i, j) /= _std.elem(i);
			}
		}
	}

	return X;
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
