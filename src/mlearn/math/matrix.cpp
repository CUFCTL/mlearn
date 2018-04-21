/**
 * @file math/matrix.cpp
 *
 * Implementation of the matrix library.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <stdexcept>

#include <cblas.h>
#include <lapacke.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "mlearn/math/matrix.h"
#include "mlearn/math/random.h"
#include "mlearn/util/logger.h"



namespace ML {



bool GPU = false;
int GPU_DEVICE = 0;
cublasHandle_t CUBLAS_HANDLE;
cusolverDnHandle_t CUSOLVER_HANDLE;



const float EPSILON = 1e-16;



/**
 * Throw an exception if a condition is false.
 *
 * @param condition
 * @param errmsg
 */
inline void throw_on_fail(bool condition, const std::string& errmsg="")
{
	if ( !condition ) {
		throw std::runtime_error(errmsg);
	}
}



/**
 * Determine whether a Matrix is a vector.
 *
 * @param v
 */
inline bool is_vector(const Matrix& v)
{
	return (v.rows() == 1 || v.cols() == 1);
}



/**
 * Determine whether a Matrix is square.
 *
 * @param M
 */
inline bool is_square(const Matrix& M)
{
	return (M.rows() == M.cols());
}



/**
 * Determine the length of a vector.
 *
 * @param v
 */
inline int length(const Matrix& v)
{
	return (v.rows() == 1) ? v.cols() : v.rows();
}



/**
 * Create a connection to the GPU.
 */
void gpu_init()
{
	if ( !GPU ) {
		return;
	}

	cublasStatus_t cublas_status = cublasCreate(&CUBLAS_HANDLE);
	cusolverStatus_t cusolver_status = cusolverDnCreate(&CUSOLVER_HANDLE);

	throw_on_fail(
		cublas_status == CUBLAS_STATUS_SUCCESS && cusolver_status == CUSOLVER_STATUS_SUCCESS,
		"Failed to initialize GPU");
}



/**
 * Close the connection to the GPU.
 */
void gpu_finalize()
{
	if ( !GPU ) {
		return;
	}

	cublasStatus_t cublas_status = cublasDestroy(CUBLAS_HANDLE);
	cusolverStatus_t cusolver_status = cusolverDnDestroy(CUSOLVER_HANDLE);

	throw_on_fail(
		cublas_status == CUBLAS_STATUS_SUCCESS && cusolver_status == CUSOLVER_STATUS_SUCCESS,
		"Failed to finalize GPU");
}



/**
 * Allocate memory on the GPU.
 *
 * @param size
 */
void * gpu_malloc(size_t size)
{
	if ( !GPU ) {
		return nullptr;
	}

	void *ptr = nullptr;

	cudaError_t error = cudaMalloc(&ptr, size);
	throw_on_fail(error == cudaSuccess, "Failed to allocate GPU memory");

	return ptr;
}



/**
 * Free memory on the GPU.
 *
 * @param ptr
 */
void gpu_free(void *ptr)
{
	if ( !GPU ) {
		return;
	}

	cudaError_t error = cudaFree(ptr);
	throw_on_fail(error == cudaSuccess, "Failed to release GPU memory");
}



/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 */
Matrix::Matrix(int rows, int cols)
{
	Logger::log(LogLevel::Debug, "debug: new Matrix(%d, %d)",
		rows, cols);

	_rows = rows;
	_cols = cols;
	_data_cpu = new float[rows * cols];
	_data_gpu = (float *)gpu_malloc(rows * cols * sizeof(float));
	_transposed = false;
	_T = new Matrix();

	// initialize transpose
	_T->_rows = rows;
	_T->_cols = cols;
	_T->_data_cpu = _data_cpu;
	_T->_data_gpu = _data_gpu;
	_T->_transposed = true;
	_T->_T = nullptr;
}



/**
 * Construct a matrix with arbitrary data.
 *
 * @param rows
 * @param cols
 * @param data
 */
Matrix::Matrix(int rows, int cols, float *data)
	: Matrix(rows, cols)
{
	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			elem(i, j) = data[i * cols + j];
		}
	}

	gpu_write();
}



/**
 * Copy a range of columns in a matrix.
 *
 * @param M
 * @param i
 * @param j
 */
Matrix::Matrix(const Matrix& M, int i, int j)
	: Matrix(M._rows, j - i)
{
	Logger::log(LogLevel::Debug, "debug: C [%d,%d] <- M(:, %d:%d) [%d,%d]",
		_rows, _cols,
		i + 1, j, M._rows, j - i);

	assert(0 <= i && i < j && j <= M._cols);

	memcpy(_data_cpu, &ELEM(M, 0, i), _rows * _cols * sizeof(float));

	gpu_write();
}



/**
 * Copy-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(const Matrix& M)
	: Matrix(M, 0, M._cols)
{
}



/**
 * Move-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(Matrix&& M)
	: Matrix()
{
	swap(*this, M);
}



/**
 * Construct an empty matrix.
 */
Matrix::Matrix()
{
	_rows = 0;
	_cols = 0;
	_data_cpu = nullptr;
	_data_gpu = nullptr;
	_transposed = false;
	_T = nullptr;
}



/**
 * Destruct a matrix.
 */
Matrix::~Matrix()
{
	if ( _transposed ) {
		return;
	}

	delete[] _data_cpu;
	gpu_free(_data_gpu);

	delete _T;
}



/**
 * Initialize a matrix to identity.
 */
void Matrix::init_identity()
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- eye(%d)",
		M._rows, M._rows,
		M._rows);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = (i == j);
		}
	}

	M.gpu_write();
}



/**
 * Initialize a matrix to all ones.
 */
void Matrix::init_ones()
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- ones(%d, %d)",
		M._rows, M._cols,
		M._rows, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = 1;
		}
	}

	M.gpu_write();
}



/**
 * Initialize a matrix to normally-distributed random numbers.
 */
void Matrix::init_random()
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- randn(%d, %d)",
		M._rows, M._cols,
		M._rows, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = Random::normal();
		}
	}

	M.gpu_write();
}



/**
 * Initialize a matrix to all zeros.
 */
void Matrix::init_zeros()
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- zeros(%d, %d)",
		M._rows, M._cols,
		M._rows, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = 0;
		}
	}

	M.gpu_write();
}



/**
 * Construct an identity matrix.
 *
 * @param rows
 */
Matrix Matrix::identity(int rows)
{
	Matrix M(rows, rows);
	M.init_identity();

	return M;
}



/**
 * Construct a matrix of all ones.
 *
 * @param rows
 * @param cols
 */
Matrix Matrix::ones(int rows, int cols)
{
	Matrix M(rows, cols);
	M.init_ones();

	return M;
}



/**
 * Construct a matrix of normally-distributed random numbers.
 *
 * @param rows
 * @param cols
 */
Matrix Matrix::random(int rows, int cols)
{
	Matrix M(rows, cols);
	M.init_random();

	return M;
}



/**
 * Construct a zero matrix.
 *
 * @param rows
 * @param cols
 */
Matrix Matrix::zeros(int rows, int cols)
{
	Matrix M(rows, cols);
	M.init_zeros();

	return M;
}



/**
 * Save a matrix to a file.
 */
IODevice& operator<<(IODevice& file, const Matrix& M)
{
	file << M._rows;
	file << M._cols;
	file.write(reinterpret_cast<const char *>(M._data_cpu), M._rows * M._cols * sizeof(float));
	return file;
}



/**
 * Load a matrix from a file.
 */
IODevice& operator>>(IODevice& file, Matrix& M)
{
	if ( M._rows * M._cols != 0 ) {
		Logger::log(LogLevel::Error, "error: cannot load into non-empty matrix");
		exit(1);
	}

	int rows, cols;
	file >> rows;
	file >> cols;

	M = Matrix(rows, cols);
	file.read(reinterpret_cast<char *>(M._data_cpu), M._rows * M._cols * sizeof(float));
	return file;
}



/**
 * Print a matrix.
 */
void Matrix::print() const
{
	std::cout << "[" << _rows << ", " << _cols << "]\n";

	for ( int i = 0; i < _rows; i++ ) {
		for ( int j = 0; j < _cols; j++ ) {
			std::cout << std::right << std::setw(10) << std::setprecision(4) << elem(i, j);
		}
		std::cout << "\n";
	}
}



/**
 * Copy matrix data from device memory to host memory.
 */
void Matrix::gpu_read()
{
	if ( !GPU ) {
		return;
	}

	cublasStatus_t status = cublasGetMatrix(
		_rows, _cols, sizeof(float),
		_data_gpu, _rows,
		_data_cpu, _rows);
	throw_on_fail(status == CUBLAS_STATUS_SUCCESS, "Failed to write GPU memory");
}



/**
 * Copy matrix data from host memory to device memory.
 */
void Matrix::gpu_write()
{
	if ( !GPU ) {
		return;
	}

	cublasStatus_t status = cublasSetMatrix(
		_rows, _cols, sizeof(float),
		_data_cpu, _rows,
		_data_gpu, _rows);
	throw_on_fail(status == CUBLAS_STATUS_SUCCESS, "Failed to write GPU memory");
}



/**
 * Compute the determinant of a matrix using LU decomposition:
 *
 *   det(M) = det(P * L * U)
 */
float Matrix::determinant() const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: d <- det(M [%d,%d])",
		M._rows, M._cols);

	int m = M._rows;
	int n = M._cols;
	Matrix U = M;
	int lda = m;
	int lipiv = std::min(m, n);
	int *ipiv = new int[lipiv];

	// compute LU decomposition
	if ( GPU ) {
		int lwork;

		cusolverStatus_t status = cusolverDnSgetrf_bufferSize(
			CUSOLVER_HANDLE,
			m, n, U._data_gpu, lda,
			&lwork
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		float *work = (float *)gpu_malloc(lwork * sizeof(float));
		int *devIpiv = (int *)gpu_malloc(lipiv * sizeof(int));
		int *info = (int *)gpu_malloc(sizeof(int));

		status = cusolverDnSgetrf(
			CUSOLVER_HANDLE,
			m, n, U._data_gpu, lda,
			work, devIpiv,
			info);
		assert(status == CUSOLVER_STATUS_SUCCESS);
		// assert(*info >= 0);

		// TODO: handle with buffer type
		cudaError_t error = cudaMemcpy(ipiv, devIpiv, lipiv * sizeof(int), cudaMemcpyDeviceToHost);
		assert(error == cudaSuccess);

		gpu_free(work);
		gpu_free(devIpiv);
		gpu_free(info);

		U.gpu_read();
	}
	else {
		int info = LAPACKE_sgetrf_work(
			LAPACK_COL_MAJOR,
			m, n, U._data_cpu, lda,
			ipiv);
		assert(info >= 0);
	}

	// compute det(A) = det(P * L * U) = 1^S * det(U)
	float det = 1;
	for ( int i = 0; i < std::min(m, n); i++ ) {
		if ( i + 1 != ipiv[i] ) {
			det *= -1;
		}
	}

	for ( int i = 0; i < std::min(m, n); i++ ) {
		det *= U.elem(i, i);
	}

	// cleanup
	delete[] ipiv;

	return det;
}



/**
 * Compute the diagonal matrix of a vector.
 */
Matrix Matrix::diagonalize() const
{
	const Matrix& v = *this;

	Logger::log(LogLevel::Debug, "debug: D [%d,%d] <- diag(v [%d,%d])",
		length(v), length(v), v._rows, v._cols);

	assert(is_vector(v));

	int n = length(v);
	Matrix D = Matrix::zeros(n, n);

	for ( int i = 0; i < n; i++ ) {
		D.elem(i, i) = v._data_cpu[i];
	}

	D.gpu_write();

	return D;
}



/**
 * Compute the eigenvalues and eigenvectors of a symmetric matrix.
 *
 * The eigenvalues are returned as a diagonal matrix, and the
 * eigenvectors are returned as column vectors. The i-th
 * eigenvalue corresponds to the i-th column vector. The eigenvalues
 * are returned in ascending order.
 *
 * @param n1
 * @param V
 * @param D
 */
void Matrix::eigen(int n1, Matrix& V, Matrix& D) const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: V [%d,%d], D [%d,%d] <- eig(M [%d,%d], %d)",
		M._rows, n1,
		n1, n1,
		M._rows, M._cols, n1);

	assert(is_square(M));

	V = M;
	D = Matrix(1, M._cols);

	// compute eigenvalues and eigenvectors
	int n = M._cols;
	int lda = M._rows;

	if ( GPU ) {
		int lwork;

		cusolverStatus_t status = cusolverDnSsyevd_bufferSize(
			CUSOLVER_HANDLE,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, V._data_gpu, lda,
			D._data_gpu,
			&lwork
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		float *work = (float *)gpu_malloc(lwork * sizeof(float));
		int *info = (int *)gpu_malloc(sizeof(int));

		status = cusolverDnSsyevd(
			CUSOLVER_HANDLE,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, V._data_gpu, lda,
			D._data_gpu,
			work, lwork,
			info
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);
		// assert(*info >= 0);

		gpu_free(work);
		gpu_free(info);

		V.gpu_read();
		D.gpu_read();
	}
	else {
		int lwork = 3 * n;
		float *work = new float[lwork];

		int info = LAPACKE_ssyev_work(
			LAPACK_COL_MAJOR, 'V', 'U',
			n, V._data_cpu, lda,
			D._data_cpu,
			work, lwork
		);
		assert(info == 0);

		delete[] work;
	}

	// take only positive eigenvalues
	int i = 0;
	while ( i < D._cols && D.elem(0, i) < EPSILON ) {
		i++;
	}

	// take only the n1 largest eigenvalues
	i = std::max(i, D._cols - n1);

	V = V(i, V._cols);
	D = D(i, D._cols).diagonalize();
}



/**
 * Compute the inverse of a square matrix using LU decomposition.
 */
Matrix Matrix::inverse() const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M^-1 [%d,%d] <- inv(M [%d,%d])",
		M._rows, M._cols, M._rows, M._cols);

	assert(is_square(M));

	int m = M._rows;
	int n = M._cols;
	Matrix A = M;
	Matrix M_inv = Matrix::identity(n);
	int lda = M._rows;

	if ( GPU ) {
		int lwork;

		cusolverStatus_t status = cusolverDnSgetrf_bufferSize(
			CUSOLVER_HANDLE,
			m, n, A._data_gpu, lda,
			&lwork
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		float *work = (float *)gpu_malloc(lwork * sizeof(float));
		int *ipiv = (int *)gpu_malloc(n * sizeof(int));
		int *info = (int *)gpu_malloc(sizeof(int));

		status = cusolverDnSgetrf(
			CUSOLVER_HANDLE,
			m, n, A._data_gpu, lda,
			work, ipiv,
			info);
		assert(status == CUSOLVER_STATUS_SUCCESS);
		// assert(*info >= 0);

		cusolverDnSgetrs(
			CUSOLVER_HANDLE, CUBLAS_OP_N,
			n, n, A._data_gpu, lda,
			ipiv,
			M_inv._data_gpu, n,
			info);

		gpu_free(work);
		gpu_free(ipiv);
		gpu_free(info);

		// throw_on_fail(info == 0, "Failed to compute inverse");

		M_inv.gpu_read();
	}
	else {
		int *ipiv = new int[n];

		int info = LAPACKE_sgetrf_work(
			LAPACK_COL_MAJOR,
			m, n, A._data_cpu, lda,
			ipiv);
		assert(info >= 0);

		info = LAPACKE_sgetrs_work(
			LAPACK_COL_MAJOR, 'N',
			n, n, A._data_cpu, lda,
			ipiv,
			M_inv._data_cpu, n);

		delete[] ipiv;

		throw_on_fail(info == 0, "Failed to compute inverse");
	}

	return M_inv;
}



/**
 * Compute the mean column of a matrix.
 */
Matrix Matrix::mean_column() const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: mu [%d,%d] <- mean(M [%d,%d], 2)",
		M._rows, 1, M._rows, M._cols);

	Matrix mu = Matrix::zeros(M._rows, 1);

	for ( int i = 0; i < M._cols; i++ ) {
		for ( int j = 0; j < M._rows; j++ ) {
			mu.elem(j, 0) += M.elem(j, i);
		}
	}
	mu.gpu_write();

	mu /= M._cols;

	return mu;
}



/**
 * Compute the mean row of a matrix.
 */
Matrix Matrix::mean_row() const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: mu [%d,%d] <- mean(M [%d,%d], 1)",
		1, M._cols, M._rows, M._cols);

	Matrix mu = Matrix::zeros(1, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			mu.elem(0, j) += M.elem(i, j);
		}
	}
	mu.gpu_write();

	mu /= M._rows;

	return mu;
}



/**
 * Compute the product of two matrices.
 *
 * @param B
 */
Matrix Matrix::product(const Matrix& B) const
{
	const Matrix& A = *this;

	int m = A._transposed ? A._cols : A._rows;
	int n = B._transposed ? B._rows : B._cols;

	Matrix C = Matrix(m, n);
	C.gemm(1.0f, A, B, 0.0f);

	return C;
}



/**
 * Compute the sum of the elements of a vector.
 */
float Matrix::sum() const
{
	const Matrix& v = *this;

	Logger::log(LogLevel::Debug, "debug: s = sum(v [%d,%d])",
		v._rows, v._cols);

	assert(is_vector(v));

	int n = length(v);
	float sum = 0.0f;

	for ( int i = 0; i < n; i++ ) {
		sum += v._data_cpu[i];
	}

	return sum;
}



/**
 * Compute the economy-size singular value decomposition
 * of a matrix:
 *
 *   A = U * S * V'
 *
 * @param U
 * @param S
 * @param V
 */
void Matrix::svd(Matrix& U, Matrix& S, Matrix& V) const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: U, S, V <- svd(M [%d,%d])",
		M._rows, M._cols);

	int m = M._rows;
	int n = M._cols;
	int lda = m;
	int ldu = m;
	int ldvt = std::min(m, n);

	U = Matrix(ldu, std::min(m, n));
	S = Matrix(1, std::min(m, n));
	Matrix VT = Matrix(ldvt, n);

	if ( GPU ) {
		Matrix wA = M;
		int lwork;

		cusolverStatus_t status = cusolverDnSgesvd_bufferSize(
			CUSOLVER_HANDLE,
			m, n,
			&lwork);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		float *work = (float *)gpu_malloc(lwork * sizeof(float));
		int *info = (int *)gpu_malloc(sizeof(int));

		status = cusolverDnSgesvd(
			CUSOLVER_HANDLE, 'S', 'S',
			m, n, wA._data_gpu, lda,
			S._data_gpu,
			U._data_gpu, ldu,
			VT._data_gpu, ldvt,
			work, lwork,
			nullptr,
			info);
		assert(status == CUSOLVER_STATUS_SUCCESS);
		// assert(*info == 0);

		gpu_free(work);
		gpu_free(info);

		U.gpu_read();
		S.gpu_read();
		VT.gpu_read();
	}
	else {
		Matrix wA = M;
		int lwork = 5 * std::min(m, n);
		float *work = new float[lwork];

		int info = LAPACKE_sgesvd_work(
			LAPACK_COL_MAJOR, 'S', 'S',
			m, n, wA._data_cpu, lda,
			S._data_cpu,
			U._data_cpu, ldu,
			VT._data_cpu, ldvt,
			work, lwork);
		assert(info == 0);

		delete[] work;
	}

	S = S.diagonalize();
	V = VT.transpose();
}



/**
 * Compute the transpose of a matrix.
 */
Matrix Matrix::transpose() const
{
	const Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M' [%d,%d] <- transpose(M [%d,%d])",
		M._cols, M._rows, M._rows, M._cols);

	Matrix MT(M._cols, M._rows);

	for ( int i = 0; i < MT._rows; i++ ) {
		for ( int j = 0; j < MT._cols; j++ ) {
			MT.elem(i, j) = M.elem(j, i);
		}
	}

	MT.gpu_write();

	return MT;
}



/**
 * Add a matrix to another matrix.
 *
 * @param B
 */
void Matrix::add(const Matrix& B)
{
	Matrix& A = *this;

	A.axpy(1.0f, B);
}



/**
 * Assign a column of a matrix.
 *
 * @param i
 * @param B
 * @param j
 */
void Matrix::assign_column(int i, const Matrix& B, int j)
{
	Matrix& A = *this;

	Logger::log(LogLevel::Debug, "debug: A(:, %d) [%d,%d] <- B(:, %d) [%d,%d]",
		i + 1, A._rows, 1,
		j + 1, B._rows, 1);

	assert(A._rows == B._rows);
	assert(0 <= i && i < A._cols);
	assert(0 <= j && j < B._cols);

	memcpy(&ELEM(A, 0, i), B._data_cpu, B._rows * sizeof(float));

	A.gpu_write();
}



/**
 * Assign a row of a matrix.
 *
 * @param i
 * @param B
 * @param j
 */
void Matrix::assign_row(int i, const Matrix& B, int j)
{
	Matrix& A = *this;

	Logger::log(LogLevel::Debug, "debug: A(%d, :) [%d,%d] <- B(%d, :) [%d,%d]",
		i + 1, 1, A._cols,
		j + 1, 1, B._cols);

	assert(A._cols == B._cols);
	assert(0 <= i && i < A._rows);
	assert(0 <= j && j < B._rows);

	for ( int k = 0; k < A._cols; k++ ) {
		A.elem(i, k) = B.elem(j, k);
	}

	A.gpu_write();
}



/**
 * Apply a function to each element of a matrix.
 *
 * @param f
 */
void Matrix::elem_apply(elem_func_t f)
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- f(M [%d,%d])",
		M._rows, M._cols, M._rows, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = f(M.elem(i, j));
		}
	}

	M.gpu_write();
}



/**
 * Subtract a matrix from another matrix.
 *
 * @param B
 */
void Matrix::subtract(const Matrix& B)
{
	Matrix& A = *this;

	A.axpy(-1.0f, B);
}



/**
 * Subtract a column vector from each column in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - a * 1_N'
 *
 * @param a
 */
void Matrix::subtract_columns(const Matrix& a)
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- M [%d,%d] - a [%d,%d] * 1_N' [%d,%d]",
		M._rows, M._cols,
		M._rows, M._cols,
		a._rows, a._cols,
		1, M._cols);

	assert(M._rows == a._rows && a._cols == 1);

	for ( int i = 0; i < M._cols; i++ ) {
		for ( int j = 0; j < M._rows; j++ ) {
			M.elem(j, i) -= a.elem(j, 0);
		}
	}
	M.gpu_write();
}



/**
 * Subtract a row vector from each row in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - 1_N * a
 *
 * @param a
 */
void Matrix::subtract_rows(const Matrix& a)
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- M [%d,%d] - a [%d,%d] * 1_N [%d,%d]",
		M._rows, M._cols,
		M._rows, M._cols,
		M._rows, 1,
		a._rows, a._cols);

	assert(M._cols == a._cols && a._rows == 1);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) -= a.elem(0, j);
		}
	}
	M.gpu_write();
}



/**
 * Wrapper function for BLAS axpy:
 *
 *   B <- alpha * A + B
 *
 * @param alpha
 * @param A
 */
void Matrix::axpy(float alpha, const Matrix& A)
{
	Matrix& B = *this;

	Logger::log(LogLevel::Debug, "debug: B [%d,%d] <- %g * A [%d,%d] + B",
		B._rows, B._cols,
		alpha, A._rows, A._cols);

	assert(A._rows == B._rows && A._cols == B._cols);

	int n = B._rows * B._cols;
	int incX = 1;
	int incY = 1;

	if ( GPU ) {
		cublasStatus_t status = cublasSaxpy(
			CUBLAS_HANDLE, n,
			&alpha,
			A._data_gpu, incX,
			B._data_gpu, incY);
		assert(status == CUBLAS_STATUS_SUCCESS);

		B.gpu_read();
	}
	else {
		cblas_saxpy(n, alpha, A._data_cpu, incX, B._data_cpu, incY);
	}
}



/**
 * Wrapper function for BLAS dot:
 *
 *   dot <- x' * y
 *
 * @param y
 */
float Matrix::dot(const Matrix& y) const
{
	const Matrix& x = *this;

	Logger::log(LogLevel::Debug, "debug: dot <- x' [%d,%d] * y [%d,%d]",
		x._rows, x._cols, y._rows, y._cols);

	assert(is_vector(x) && is_vector(y));
	assert(length(x) == length(y));

	int n = length(x);
	int incX = 1;
	int incY = 1;
	float dot;

	if ( GPU ) {
		cublasStatus_t status = cublasSdot(
			CUBLAS_HANDLE, n,
			x._data_gpu, incX,
			y._data_gpu, incY,
			&dot);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	else {
		dot = cblas_sdot(n, x._data_cpu, incX, y._data_cpu, incY);
	}

	return dot;
}



/**
 * Wrapper function for BLAS gemm:
 *
 *   C <- alpha * A * B + beta * C
 *
 * @param alpha
 * @param A
 * @param B
 * @param beta
 */
void Matrix::gemm(float alpha, const Matrix& A, const Matrix& B, float beta)
{
	Matrix& C = *this;

	int m = A._transposed ? A._cols : A._rows;
	int k1 = A._transposed ? A._rows : A._cols;
	int k2 = B._transposed ? B._cols : B._rows;
	int n = B._transposed ? B._rows : B._cols;

	Logger::log(LogLevel::Debug, "debug: C [%d,%d] <- A%s [%d,%d] * B%s [%d,%d] + %g * C",
		C._rows, C._cols,
		A._transposed ? "'" : "", m, k1,
		B._transposed ? "'" : "", k2, n,
		beta);

	assert(C._rows == m && C._cols == n && k1 == k2);

	if ( GPU ) {
		cublasOperation_t TransA = A._transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t TransB = B._transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasStatus_t status = cublasSgemm(
			CUBLAS_HANDLE, TransA, TransB,
			m, n, k1,
			&alpha,
			A._data_gpu, A._rows,
			B._data_gpu, B._rows,
			&beta,
			C._data_gpu, C._rows);
		assert(status == CUBLAS_STATUS_SUCCESS);

		C.gpu_read();
	}
	else {
		CBLAS_TRANSPOSE TransA = A._transposed ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE TransB = B._transposed ? CblasTrans : CblasNoTrans;

		cblas_sgemm(
			CblasColMajor, TransA, TransB,
			m, n, k1,
			alpha,
			A._data_cpu, A._rows,
			B._data_cpu, B._rows,
			beta,
			C._data_cpu, C._rows);
	}
}



/**
 * Wrapper function for BLAS nrm2:
 *
 *   nrm2 <- ||x||
 */
float Matrix::nrm2() const
{
	const Matrix& x = *this;

	Logger::log(LogLevel::Debug, "debug: nrm2 <- ||x [%d,%d]||",
		x._rows, x._cols);

	assert(is_vector(x));

	int n = length(x);
	int incX = 1;
	float nrm2;

	if ( GPU ) {
		cublasStatus_t status = cublasSnrm2(
			CUBLAS_HANDLE, n,
			x._data_gpu, incX,
			&nrm2);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	else {
		nrm2 = cblas_snrm2(n, x._data_cpu, incX);
	}

	return nrm2;
}



/**
 * Wrapper function for BLAS scal:
 *
 *   M <- alpha * M
 *
 * @param alpha
 */
void Matrix::scal(float alpha)
{
	Matrix& M = *this;

	Logger::log(LogLevel::Debug, "debug: M [%d,%d] <- %g * M",
		M._rows, M._cols, alpha);

	int n = M._rows * M._cols;
	int incX = 1;

	if ( GPU ) {
		cublasStatus_t status = cublasSscal(
			CUBLAS_HANDLE, n,
			&alpha,
			M._data_gpu, incX);
		assert(status == CUBLAS_STATUS_SUCCESS);

		M.gpu_read();
	}
	else {
		cblas_sscal(n, alpha, M._data_cpu, incX);
	}
}



/**
 * Wrapper function for BLAS syr:
 *
 *   A <- alpha * x * x' + A
 *
 * @param alpha
 * @param x
 */
void Matrix::syr(float alpha, const Matrix& x)
{
	Matrix& A = *this;

	Logger::log(LogLevel::Debug, "debug: A [%d,%d] <- %g * x [%d,%d] * x' [%d,%d] + A",
		A._rows, A._cols,
		alpha, x._rows, x._cols, x._cols, x._rows);

	assert(A._rows == A._cols && is_vector(x) && A._rows == length(x));

	int n = A._rows;
	int incX = 1;

	if ( GPU ) {
		cublasStatus_t status = cublasSsyr(
			CUBLAS_HANDLE, CUBLAS_FILL_MODE_UPPER,
			n, &alpha,
			x._data_gpu, incX,
			A._data_gpu, A._rows);
		assert(status == CUBLAS_STATUS_SUCCESS);

		A.gpu_read();
	}
	else {
		cblas_ssyr(
			CblasColMajor, CblasUpper,
			n, alpha,
			x._data_cpu, incX,
			A._data_cpu, A._rows);
	}
}



/**
 * Wrapper function for BLAS syrk:
 *
 *   C <- alpha * A * A' + beta * C
 *   C <- alpha * A' * A + beta * C
 *
 * @param trans
 * @param alpha
 * @param A
 * @param beta
 */
void Matrix::syrk(bool trans, float alpha, const Matrix& A, float beta)
{
	Matrix& C = *this;

	int n = trans ? A._cols : A._rows;
	int k = trans ? A._rows : A._cols;

	Logger::log(LogLevel::Debug, "debug: C [%d,%d] <- %g * A%s [%d,%d] * A%s [%d,%d] + %g * C",
		C._rows, C._cols,
		alpha,
		trans ? "'" : "", n, k,
		trans ? "" : "'", k, n,
		beta);

	assert(C._rows == C._cols && C._rows == n);

	if ( GPU ) {
		cublasOperation_t Trans = trans ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasStatus_t status = cublasSsyrk(
			CUBLAS_HANDLE, CUBLAS_FILL_MODE_UPPER, Trans,
			n, k, &alpha,
			A._data_gpu, A._rows,
			&beta,
			C._data_gpu, C._rows);
		assert(status == CUBLAS_STATUS_SUCCESS);

		C.gpu_read();
	}
	else {
		CBLAS_TRANSPOSE Trans = trans ? CblasTrans : CblasNoTrans;

		cblas_ssyrk(
			CblasColMajor, CblasUpper, Trans,
			n, k, alpha,
			A._data_cpu, A._rows,
			beta,
			C._data_cpu, C._rows);
	}
}



/**
 * Swap function for Matrix.
 *
 * @param A
 * @param B
 */
void swap(Matrix& A, Matrix& B)
{
	std::swap(A._rows, B._rows);
	std::swap(A._cols, B._cols);
	std::swap(A._data_cpu, B._data_cpu);
	std::swap(A._data_gpu, B._data_gpu);
	std::swap(A._transposed, B._transposed);
	std::swap(A._T, B._T);
}



}
