/**
 * @file math/matrix_utils.cu
 *
 * Library of helpful matrix functions.
 */
#include <cassert>
#include <cmath>
#include "mlearn/cuda/device.h"
#include "mlearn/math/matrix_utils.h"
#include "mlearn/math/random.h"



namespace mlearn {



__global__
void m_dist_COS_kernel(
	const float *x,
	const float *y,
	int n,
	float *x_dot_y,
	float *abs_x,
	float *abs_y,
	float *similarity)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if ( i >= n )
	{
		return;
	}

	// compute x * y, ||x|| and ||y||
	x_dot_y[i] = x[i] * y[i];
	abs_x[i] = x[i] * x[i];
	abs_y[i] = y[i] * y[i];

	__syncthreads();

	for ( int p = 2; p <= n; p *= 2 )
	{
		if ( i % p == 0 )
		{
			x_dot_y[i] += x_dot_y[i + p/2];
			abs_x[i] += abs_x[i + p/2];
			abs_y[i] += abs_y[i + p/2];
		}

		__syncthreads();
	}

	// compute similarity
	if ( i == 0 )
	{
		*similarity = x_dot_y[0] / sqrt(abs_x[0] * abs_y[0]);
	}
}



/**
 * Compute the COS distance between two column vectors.
 *
 * Cosine similarity is the cosine of the angle between x and y:
 * S_cos(x, y) = x * y / (||x|| * ||y||)
 *
 * Since S_cos is on [-1, 1], we transform S_cos to be on [0, 2]:
 * d_cos(x, y) = 1 - S_cos(x, y)
 *
 * @param A
 * @param i
 * @param B
 * @param j
 */
float m_dist_COS(const Matrix& A, int i, const Matrix& B, int j)
{
	assert(A.rows() == B.rows());
	assert(0 <= i && i < A.cols() && 0 <= j && j < B.cols());

	if ( Device::instance() )
	{
		// compute similarity
		Buffer<float> x_dot_y(A.rows());
		Buffer<float> abs_x(A.rows());
		Buffer<float> abs_y(A.rows());
		Buffer<float> similarity(1);

		const int BLOCK_SIZE = 256;
		const int GRID_SIZE = (A.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE;
		m_dist_COS_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			&A.buffer().device_data()[i * A.rows()],
			&B.buffer().device_data()[j * B.rows()],
			A.rows(),
			x_dot_y.device_data(),
			abs_x.device_data(),
			abs_y.device_data(),
			similarity.device_data()
		);
		CHECK_CUDA(cudaGetLastError());

		similarity.read();

		// compute distance
		return 1 - similarity.host_data()[0];
	}
	else
	{
		// compute x * y, ||x|| and ||y||
		float x_dot_y = 0;
		float abs_x = 0;
		float abs_y = 0;

		for ( int k = 0; k < A.rows(); k++ )
		{
			x_dot_y += A.elem(k, i) * B.elem(k, j);
			abs_x += A.elem(k, i) * A.elem(k, i);
			abs_y += B.elem(k, j) * B.elem(k, j);
		}

		// compute similarity
		float similarity = x_dot_y / sqrt(abs_x * abs_y);

		// compute distance
		return 1 - similarity;
	}
}



__global__
void m_dist_L1_kernel(
	const float *x,
	const float *y,
	int n,
	float *dist)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if ( i >= n )
	{
		return;
	}

	dist[i] = fabs(x[i] - y[i]);

	__syncthreads();

	for ( int p = 2; p <= n; p *= 2 )
	{
		if ( i % p == 0 )
		{
			dist[i] += dist[i + p/2];
		}

		__syncthreads();
	}
}



/**
 * Compute the L1 distance between two column vectors.
 *
 * L1 is the Taxicab distance:
 * d_L1(x, y) = |x - y|
 *
 * @param A
 * @param i
 * @param B
 * @param j
 */
float m_dist_L1(const Matrix& A, int i, const Matrix& B, int j)
{
	assert(A.rows() == B.rows());
	assert(0 <= i && i < A.cols() && 0 <= j && j < B.cols());

	if ( Device::instance() )
	{
		Buffer<float> dist(A.rows());

		const int BLOCK_SIZE = 256;
		const int GRID_SIZE = (A.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE;
		m_dist_L1_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			&A.buffer().device_data()[i * A.rows()],
			&B.buffer().device_data()[j * B.rows()],
			A.rows(),
			dist.device_data()
		);
		CHECK_CUDA(cudaGetLastError());

		dist.read(1);

		return dist.host_data()[0];
	}
	else
	{
		float dist = 0;

		for ( int k = 0; k < A.rows(); k++ ) {
			dist += fabs(A.elem(k, i) - B.elem(k, j));
		}

		return dist;
	}
}



__global__
void m_dist_L2_kernel(
	const float *x,
	const float *y,
	int n,
	float *dist)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if ( i >= n )
	{
		return;
	}

	dist[i] = (x[i] - y[i]) * (x[i] - y[i]);

	__syncthreads();

	for ( int p = 2; p <= n; p *= 2 )
	{
		if ( i % p == 0 )
		{
			dist[i] += dist[i + p/2];
		}

		__syncthreads();
	}
}



/**
 * Compute the L2 distance between two column vectors.
 *
 * L2 is the Euclidean distance:
 * d_L2(x, y) = ||x - y||
 *
 * @param A
 * @param i
 * @param B
 * @param j
 */
float m_dist_L2(const Matrix& A, int i, const Matrix& B, int j)
{
	assert(A.rows() == B.rows());
	assert(0 <= i && i < A.cols() && 0 <= j && j < B.cols());

	if ( Device::instance() )
	{
		Buffer<float> dist(A.rows());

		const int BLOCK_SIZE = 256;
		const int GRID_SIZE = (A.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE;
		m_dist_L2_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			&A.buffer().device_data()[i * A.rows()],
			&B.buffer().device_data()[j * B.rows()],
			A.rows(),
			dist.device_data()
		);
		CHECK_CUDA(cudaGetLastError());

		dist.read(1);

		return dist.host_data()[0];
	}
	else
	{
		float dist = 0;

		for ( int k = 0; k < A.rows(); k++ ) {
			float diff = A.elem(k, i) - B.elem(k, j);
			dist += diff * diff;
		}

		dist = sqrt(dist);

		return dist;
	}
}



/**
 * Compute the mean of a list of column vectors.
 *
 * @param X
 */
Matrix m_mean(const std::vector<Matrix>& X)
{
	Matrix mu = Matrix::zeros(X[0].rows(), 1);

	for ( const Matrix& x_i : X ) {
		mu += x_i;
	}
	mu /= X.size();

	return mu;
}



/**
 * Copy a matrix X into a list of column vectors.
 *
 * @param X
 */
std::vector<Matrix> m_copy_columns(const Matrix& X)
{
	std::vector<Matrix> X_col;
	X_col.reserve(X.cols());

	for ( int i = 0; i < X.cols(); i++ ) {
		X_col.push_back(X(i));
	}

	return X_col;
}



/**
 * Subtract a vector mu from each vector in X.
 *
 * @param X
 * @param mu
 */
std::vector<Matrix> m_subtract_mean(const std::vector<Matrix>& X, const Matrix& mu)
{
	std::vector<Matrix> X_sub;
	X_sub.reserve(X.size());

	for ( const Matrix& x_i : X ) {
		X_sub.push_back(x_i - mu);
	}

	return X_sub;
}



/**
 * Copy a matrix X into a list X_c of class
 * submatrices.
 *
 * This function assumes that the columns in X
 * are grouped by class.
 *
 * @param X
 * @param y
 * @param c
 */
std::vector<Matrix> m_copy_classes(const Matrix& X, const std::vector<int>& y, int c)
{
	std::vector<Matrix> X_c;
	X_c.reserve(c);

	int i, j;
	for ( i = 0, j = 0; i < c; i++ ) {
		int k = j;
		while ( k < X.cols() && y[k] == y[j] ) {
			k++;
		}

		X_c.push_back(X(j, k));
		j = k;
	}

	assert(j == X.cols());

	return X_c;
}



/**
 * Compute the mean of each class for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * @param X_c
 */
std::vector<Matrix> m_class_means(const std::vector<Matrix>& X_c)
{
	std::vector<Matrix> U;
	U.reserve(X_c.size());

	for ( const Matrix& X_c_i : X_c ) {
		U.push_back(X_c_i.mean_column());
	}

	return U;
}



/**
 * Compute the class covariance matrices for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_i = (X_c_i - U_i) * (X_c_i - U_i)'
 *
 * @param X_c
 * @param U
 */
std::vector<Matrix> m_class_scatters(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U)
{
	std::vector<Matrix> S;
	S.reserve(X_c.size());

	for ( size_t i = 0; i < X_c.size(); i++ ) {
		Matrix X_c_i = X_c[i];
		X_c_i.subtract_columns(U[i]);

		S.push_back(X_c_i * X_c_i.T());
	}

	return S;
}



/**
 * Compute the between-scatter matrix S_b for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_b = sum(n_i * (u_i - u) * (u_i - u)', i=1:c),
 *
 * @param X_c
 * @param U
 */
Matrix m_scatter_between(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U)
{
	int N = X_c[0].rows();

	// compute the mean of all classes
	Matrix u = m_mean(U);

	// compute the between-scatter S_b
	Matrix S_b = Matrix::zeros(N, N);

	for ( size_t i = 0; i < X_c.size(); i++ ) {
		Matrix U_i = U[i] - u;

		S_b.gemm(X_c[i].cols(), U_i, U_i.T(), 1.0f);
	}

	return S_b;
}



/**
 * Compute the within-scatter matrix S_w for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_w = sum((X_c_i - U_i) * (X_c_i - U_i)', i=1:c)
 *
 * @param X_c
 * @param U
 */
Matrix m_scatter_within(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U)
{
	int N = U[0].rows();
	Matrix S_w = Matrix::zeros(N, N);

	for ( size_t i = 0; i < X_c.size(); i++ ) {
		Matrix X_c_i = X_c[i];
		X_c_i.subtract_columns(U[i]);

		S_w.gemm(1.0f, X_c_i, X_c_i.T(), 1.0f);
	}

	return S_w;
}



}
