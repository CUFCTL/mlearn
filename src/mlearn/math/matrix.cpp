/**
 * @file math/matrix.cpp
 *
 * Implementation of the matrix type.
 */
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

#include "mlearn/cuda/device.h"
#include "mlearn/math/matrix.h"
#include "mlearn/math/random.h"
#include "mlearn/util/logger.h"



namespace ML {



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
	_buffer = Buffer<float>(rows * cols);
	_transposed = false;
	_T = new Matrix();

	// initialize transpose
	_T->_rows = rows;
	_T->_cols = cols;
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

	memcpy(_buffer.host_data(), &M.elem(0, i), _rows * _cols * sizeof(float));

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
	file.write(reinterpret_cast<const char *>(M._buffer.host_data()), M._rows * M._cols * sizeof(float));
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
	file.read(reinterpret_cast<char *>(M._buffer.host_data()), M._rows * M._cols * sizeof(float));
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
	Buffer<int> ipiv(std::min(m, n));

	// compute LU decomposition
	getrf(U, ipiv);

	U.gpu_read();
	ipiv.read();

	// compute det(A) = det(P * L * U) = 1^S * det(U)
	float det = 1;
	for ( int i = 0; i < std::min(m, n); i++ ) {
		if ( i + 1 != ipiv.host_data()[i] ) {
			det *= -1;
		}
	}

	for ( int i = 0; i < std::min(m, n); i++ ) {
		det *= U.elem(i, i);
	}

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
		D.elem(i, i) = v._buffer.host_data()[i];
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

	V = M;
	D = Matrix(1, M._cols);

	// compute eigenvalues and eigenvectors
	syev(V, D);

	V.gpu_read();
	D.gpu_read();

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

	int n = M._cols;
	Matrix A = M;
	Matrix M_inv = Matrix::identity(n);
	Buffer<int> ipiv(n);

	// compute LU decomposition
	getrf(A, ipiv);

	// compute inverse
	bool success = getrs(A, M_inv, ipiv);

	throw_on_fail(success, "Failed to compute inverse");

	M_inv.gpu_read();

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
		sum += v._buffer.host_data()[i];
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

	U = Matrix(m, std::min(m, n));
	S = Matrix(1, std::min(m, n));
	Matrix VT = Matrix(std::min(m, n), n);

	gesvd(U, S, VT);

	U.gpu_read();
	S.gpu_read();
	VT.gpu_read();

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

	memcpy(&A.elem(0, i), B._buffer.host_data(), B._rows * sizeof(float));

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

	if ( Device::instance() ) {
		cublasStatus_t status = cublasSaxpy(
			Device::instance()->cublas_handle(), n,
			&alpha,
			A._buffer.device_data(), incX,
			B._buffer.device_data(), incY);
		assert(status == CUBLAS_STATUS_SUCCESS);

		B.gpu_read();
	}
	else {
		cblas_saxpy(n, alpha, A._buffer.host_data(), incX, B._buffer.host_data(), incY);
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

	if ( Device::instance() ) {
		cublasStatus_t status = cublasSdot(
			Device::instance()->cublas_handle(), n,
			x._buffer.device_data(), incX,
			y._buffer.device_data(), incY,
			&dot);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	else {
		dot = cblas_sdot(n, x._buffer.host_data(), incX, y._buffer.host_data(), incY);
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

	if ( Device::instance() ) {
		cublasOperation_t TransA = A._transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t TransB = B._transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasStatus_t status = cublasSgemm(
			Device::instance()->cublas_handle(), TransA, TransB,
			m, n, k1,
			&alpha,
			A._buffer.device_data(), A._rows,
			B._buffer.device_data(), B._rows,
			&beta,
			C._buffer.device_data(), C._rows);
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
			A._buffer.host_data(), A._rows,
			B._buffer.host_data(), B._rows,
			beta,
			C._buffer.host_data(), C._rows);
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

	if ( Device::instance() ) {
		cublasStatus_t status = cublasSnrm2(
			Device::instance()->cublas_handle(), n,
			x._buffer.device_data(), incX,
			&nrm2);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	else {
		nrm2 = cblas_snrm2(n, x._buffer.host_data(), incX);
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

	if ( Device::instance() ) {
		cublasStatus_t status = cublasSscal(
			Device::instance()->cublas_handle(), n,
			&alpha,
			M._buffer.device_data(), incX);
		assert(status == CUBLAS_STATUS_SUCCESS);

		M.gpu_read();
	}
	else {
		cblas_sscal(n, alpha, M._buffer.host_data(), incX);
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

	assert(is_square(A) && is_vector(x) && A._rows == length(x));

	int n = A._rows;
	int incX = 1;

	if ( Device::instance() ) {
		cublasStatus_t status = cublasSsyr(
			Device::instance()->cublas_handle(), CUBLAS_FILL_MODE_UPPER,
			n, &alpha,
			x._buffer.device_data(), incX,
			A._buffer.device_data(), A._rows);
		assert(status == CUBLAS_STATUS_SUCCESS);

		A.gpu_read();
	}
	else {
		cblas_ssyr(
			CblasColMajor, CblasUpper,
			n, alpha,
			x._buffer.host_data(), incX,
			A._buffer.host_data(), A._rows);
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

	assert(is_square(C) && C._rows == n);

	if ( Device::instance() ) {
		cublasOperation_t Trans = trans ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasStatus_t status = cublasSsyrk(
			Device::instance()->cublas_handle(), CUBLAS_FILL_MODE_UPPER, Trans,
			n, k, &alpha,
			A._buffer.device_data(), A._rows,
			&beta,
			C._buffer.device_data(), C._rows);
		assert(status == CUBLAS_STATUS_SUCCESS);

		C.gpu_read();
	}
	else {
		CBLAS_TRANSPOSE Trans = trans ? CblasTrans : CblasNoTrans;

		cblas_ssyrk(
			CblasColMajor, CblasUpper, Trans,
			n, k, alpha,
			A._buffer.host_data(), A._rows,
			beta,
			C._buffer.host_data(), C._rows);
	}
}



/**
 * Wrapper function for LAPACK gesvd:
 *
 *   A = U * S * V^T
 *
 * @param U
 * @param S
 * @param VT
 */
void Matrix::gesvd(Matrix& U, Matrix& S, Matrix& VT) const
{
	const Matrix& A = *this;

	int m = A._rows;
	int n = A._cols;
	int lda = m;
	int ldu = m;
	int ldvt = VT._rows;

	if ( Device::instance() ) {
		Matrix wA = A;
		int lwork;

		cusolverStatus_t status = cusolverDnSgesvd_bufferSize(
			Device::instance()->cusolver_handle(),
			m, n,
			&lwork);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		Buffer<float> work(lwork, false);
		Buffer<int> info(1);

		status = cusolverDnSgesvd(
			Device::instance()->cusolver_handle(), 'S', 'S',
			m, n, wA._buffer.device_data(), lda,
			S._buffer.device_data(),
			U._buffer.device_data(), ldu,
			VT._buffer.device_data(), ldvt,
			work.device_data(), lwork,
			nullptr,
			info.device_data());
		assert(status == CUSOLVER_STATUS_SUCCESS);

		info.read();
		assert(info.host_data()[0] == 0);
	}
	else {
		Matrix wA = A;
		int lwork = 5 * std::min(m, n);
		Buffer<float> work(lwork);

		int info = LAPACKE_sgesvd_work(
			LAPACK_COL_MAJOR, 'S', 'S',
			m, n, wA._buffer.host_data(), lda,
			S._buffer.host_data(),
			U._buffer.host_data(), ldu,
			VT._buffer.host_data(), ldvt,
			work.host_data(), lwork);
		assert(info == 0);
	}
}



/**
 * Wrapper function for LAPACK getrf:
 *
 *   A = P * L * U
 *
 * @param U
 * @param ipiv
 */
void Matrix::getrf(Matrix& U, Buffer<int>& ipiv) const
{
	const Matrix& A = *this;

	int m = A._rows;
	int n = A._cols;
	int lda = m;

	if ( Device::instance() ) {
		int lwork;

		cusolverStatus_t status = cusolverDnSgetrf_bufferSize(
			Device::instance()->cusolver_handle(),
			m, n, U._buffer.device_data(), lda,
			&lwork
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		Buffer<float> work(lwork, false);
		Buffer<int> info(1);

		status = cusolverDnSgetrf(
			Device::instance()->cusolver_handle(),
			m, n, U._buffer.device_data(), lda,
			work.device_data(), ipiv.device_data(),
			info.device_data());
		assert(status == CUSOLVER_STATUS_SUCCESS);

		info.read();
		assert(info.host_data()[0] >= 0);
	}
	else {
		int info = LAPACKE_sgetrf_work(
			LAPACK_COL_MAJOR,
			m, n, U._buffer.host_data(), lda,
			ipiv.host_data());
		assert(info >= 0);
	}
}



/**
 * Wrapper function for LAPACK getrs:
 *
 *   A * X = B
 *
 * @param U
 * @param ipiv
 */
bool Matrix::getrs(const Matrix& A, Matrix& B, Buffer<int>& ipiv) const
{
	assert(is_square(A));

	int n = A._cols;
	int lda = A._rows;

	if ( Device::instance() ) {
		Buffer<int> info(1);

		cusolverStatus_t status = cusolverDnSgetrs(
			Device::instance()->cusolver_handle(), CUBLAS_OP_N,
			n, n, A._buffer.device_data(), lda,
			ipiv.device_data(),
			B._buffer.device_data(), n,
			info.device_data());
		assert(status == CUSOLVER_STATUS_SUCCESS);

		info.read();
		return (info.host_data()[0] == 0);
	}
	else {
		int info = LAPACKE_sgetrs_work(
			LAPACK_COL_MAJOR, 'N',
			n, n, A._buffer.host_data(), lda,
			ipiv.host_data(),
			B._buffer.host_data(), n);

		return (info == 0);
	}
}



/**
 * Wrapper function for LAPACK syev:
 *
 *   A = V * D * V^-1
 *
 * @param V
 * @param D
 */
void Matrix::syev(Matrix& V, Matrix& D) const
{
	const Matrix& A = *this;

	assert(is_square(A));

	int n = A._cols;
	int lda = A._rows;

	if ( Device::instance() ) {
		int lwork;

		cusolverStatus_t status = cusolverDnSsyevd_bufferSize(
			Device::instance()->cusolver_handle(),
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, V._buffer.device_data(), lda,
			D._buffer.device_data(),
			&lwork
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		Buffer<float> work(lwork, false);
		Buffer<int> info(1);

		status = cusolverDnSsyevd(
			Device::instance()->cusolver_handle(),
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_UPPER,
			n, V._buffer.device_data(), lda,
			D._buffer.device_data(),
			work.device_data(), lwork,
			info.device_data()
		);
		assert(status == CUSOLVER_STATUS_SUCCESS);

		info.read();
		assert(info.host_data()[0] == 0);
	}
	else {
		int lwork = 3 * n;
		Buffer<float> work(lwork);

		int info = LAPACKE_ssyev_work(
			LAPACK_COL_MAJOR, 'V', 'U',
			n, V._buffer.host_data(), lda,
			D._buffer.host_data(),
			work.host_data(), lwork
		);
		assert(info == 0);
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
	std::swap(A._buffer, B._buffer);
	std::swap(A._transposed, B._transposed);
	std::swap(A._T, B._T);
}



}
