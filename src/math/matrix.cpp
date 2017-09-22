/**
 * @file math/matrix.cpp
 *
 * Implementation of the matrix library.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include "math/matrix.h"
#include "math/math_utils.h"
#include "util/logger.h"

#include <cuda_runtime.h>
#include "magma_v2.h"
#include <cblas.h>
#include <lapacke.h>

namespace ML {

bool GPU = false;
int GPU_DEVICE = 0;

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

	magma_int_t stat = magma_init();
	throw_on_fail(stat == MAGMA_SUCCESS, "Failed to initialize GPU");
}

/**
 * Close the connection to the GPU.
 */
void gpu_finalize()
{
	if ( !GPU ) {
		return;
	}

	magma_int_t stat = magma_finalize();
	throw_on_fail(stat == MAGMA_SUCCESS, "Failed to finalize GPU");
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

	int stat = magma_malloc(&ptr, size);
	throw_on_fail(stat == MAGMA_SUCCESS, "Failed to allocate GPU memory");

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

	int stat = magma_free(ptr);
	throw_on_fail(stat == MAGMA_SUCCESS, "Failed to release GPU memory");
}

/**
 * Allocate a matrix on the GPU.
 *
 * @param rows
 * @param cols
 */
float * gpu_malloc_matrix(int rows, int cols)
{
	return (float *)gpu_malloc(rows * cols * sizeof(float));
}

/**
 * Get a MAGMA queue.
 */
magma_queue_t magma_queue()
{
	static int init = 1;
	static magma_queue_t queue;

	if ( GPU && init == 1 ) {
		magma_queue_create(GPU_DEVICE, &queue);
		init = 0;
	}

	return queue;
}

/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 */
Matrix::Matrix(int rows, int cols)
{
	log(LL_DEBUG, "debug: new Matrix(%d, %d)",
		rows, cols);

	this->_rows = rows;
	this->_cols = cols;
	this->_data_cpu = new float[rows * cols];
	this->_data_gpu = gpu_malloc_matrix(rows, cols);
	this->_transposed = false;
	this->_T = new Matrix();

	// initialize transpose
	this->_T->_rows = rows;
	this->_T->_cols = cols;
	this->_T->_data_cpu = this->_data_cpu;
	this->_T->_data_gpu = this->_data_gpu;
	this->_T->_transposed = true;
	this->_T->_T = nullptr;
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
			this->elem(i, j) = data[i * cols + j];
		}
	}

	this->gpu_write();
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
	log(LL_DEBUG, "debug: C [%d,%d] <- M(:, %d:%d) [%d,%d]",
		this->_rows, this->_cols,
		i + 1, j, M._rows, j - i);

	throw_on_fail(0 <= i && i < j && j <= M._cols);

	memcpy(this->_data_cpu, &ELEM(M, 0, i), this->_rows * this->_cols * sizeof(float));

	this->gpu_write();
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
	this->_rows = 0;
	this->_cols = 0;
	this->_data_cpu = nullptr;
	this->_data_gpu = nullptr;
	this->_transposed = false;
	this->_T = nullptr;
}

/**
 * Destruct a matrix.
 */
Matrix::~Matrix()
{
	if ( this->_transposed ) {
		return;
	}

	delete[] this->_data_cpu;
	gpu_free(this->_data_gpu);

	delete this->_T;
}

/**
 * Construct an identity matrix.
 *
 * @param rows
 */
Matrix Matrix::identity(int rows)
{
	log(LL_DEBUG, "debug: M [%d,%d] <- eye(%d)",
		rows, rows,
		rows);

	Matrix M(rows, rows);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < rows; j++ ) {
			M.elem(i, j) = (i == j);
		}
	}

	M.gpu_write();

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
	log(LL_DEBUG, "debug: M [%d,%d] <- ones(%d, %d)",
		rows, cols,
		rows, cols);

	Matrix M(rows, cols);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			M.elem(i, j) = 1;
		}
	}

	M.gpu_write();

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
	log(LL_DEBUG, "debug: M [%d,%d] <- randn(%d, %d)",
		rows, cols,
		rows, cols);

	Matrix M(rows, cols);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			M.elem(i, j) = RNG_normal();
		}
	}

	M.gpu_write();

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

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			M.elem(i, j) = 0;
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Print a matrix.
 *
 * @param os
 */
void Matrix::print(std::ostream& os) const
{
	os << "[" << this->_rows << ", " << this->_cols << "]\n";

	for ( int i = 0; i < this->_rows; i++ ) {
		for ( int j = 0; j < this->_cols; j++ ) {
			os << std::right << std::setw(10) << std::setprecision(4) << this->elem(i, j);
		}
		os << "\n";
	}
}

/**
 * Save a matrix to a file.
 *
 * @param file
 */
void Matrix::save(std::ofstream& file) const
{
	file.write(reinterpret_cast<const char *>(&this->_rows), sizeof(int));
	file.write(reinterpret_cast<const char *>(&this->_cols), sizeof(int));
	file.write(reinterpret_cast<const char *>(this->_data_cpu), this->_rows * this->_cols * sizeof(float));
}

/**
 * Load a matrix from a file.
 *
 * @param file
 */
void Matrix::load(std::ifstream& file)
{
	if ( this->_rows * this->_cols != 0 ) {
		log(LL_ERROR, "error: cannot load into non-empty matrix");
		exit(1);
	}

	int rows, cols;
	file.read(reinterpret_cast<char *>(&rows), sizeof(int));
	file.read(reinterpret_cast<char *>(&cols), sizeof(int));

	*this = Matrix(rows, cols);
	file.read(reinterpret_cast<char *>(this->_data_cpu), this->_rows * this->_cols * sizeof(float));
}

/**
 * Copy matrix data from device memory to host memory.
 */
void Matrix::gpu_read()
{
	if ( !GPU ) {
		return;
	}

	magma_queue_t queue = magma_queue();

	magma_getmatrix(this->_rows, this->_cols, sizeof(float),
		this->_data_gpu, this->_rows,
		this->_data_cpu, this->_rows,
		queue);
}

/**
 * Copy matrix data from host memory to device memory.
 */
void Matrix::gpu_write()
{
	if ( !GPU ) {
		return;
	}

	magma_queue_t queue = magma_queue();

	magma_setmatrix(this->_rows, this->_cols, sizeof(float),
		this->_data_cpu, this->_rows,
		this->_data_gpu, this->_rows,
		queue);
}

/**
 * Compute the determinant of a matrix using LU decomposition:
 *
 *   det(M) = det(P * L * U)
 */
float Matrix::determinant() const
{
	const Matrix& M = *this;

	log(LL_DEBUG, "debug: d <- det(M [%d,%d])",
		M._rows, M._cols);

	int m = M._rows;
	int n = M._cols;
	Matrix U = M;
	int lda = m;
	int *ipiv = new int[min(m, n)];

	// compute LU decomposition
	if ( GPU ) {
		int info;

		magma_sgetrf_gpu(
			m, n, U._data_gpu, lda,
			ipiv,
			&info);
		throw_on_fail(info >= 0);

		U.gpu_read();
	}
	else {
		int info = LAPACKE_sgetrf_work(
			LAPACK_COL_MAJOR,
			m, n, U._data_cpu, lda,
			ipiv);
		throw_on_fail(info >= 0);
	}

	// compute det(A) = det(P * L * U) = 1^S * det(U)
	float det = 1;
	for ( int i = 0; i < min(m, n); i++ ) {
		if ( i + 1 != ipiv[i] ) {
			det *= -1;
		}
	}

	for ( int i = 0; i < min(m, n); i++ ) {
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

	log(LL_DEBUG, "debug: D [%d,%d] <- diag(v [%d,%d])",
		length(v), length(v), v._rows, v._cols);

	throw_on_fail(is_vector(v));

	int n = length(v);
	Matrix D = Matrix::zeros(n, n);

	for ( int i = 0; i < n; i++ ) {
		D.elem(i, i) = v._data_cpu[i];
	}

	D.gpu_write();

	return D;
}

/**
 * Compute the dot product of two vectors.
 *
 * @param b
 */
float Matrix::dot(const Matrix& b) const
{
	const Matrix& a = *this;

	log(LL_DEBUG, "debug: d = dot(a [%d,%d], b [%d,%d])",
		a._rows, a._cols, b._rows, b._cols);

	throw_on_fail(is_vector(a) && is_vector(b));
	throw_on_fail(length(a) == length(b));

	int n = length(a);
	int incX = 1;
	int incY = 1;
	float dot;

	if ( GPU ) {
		magma_queue_t queue = magma_queue();

		dot = magma_sdot(n, a._data_gpu, incX, b._data_gpu, incY, queue);
	}
	else {
		dot = cblas_sdot(n, a._data_cpu, incX, b._data_cpu, incY);
	}

	return dot;
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

	log(LL_DEBUG, "debug: V [%d,%d], D [%d,%d] <- eig(M [%d,%d], %d)",
		M._rows, n1,
		n1, n1,
		M._rows, M._cols, n1);

	throw_on_fail(is_square(M));

	V = M;
	D = Matrix(1, M._cols);

	// compute eigenvalues and eigenvectors
	int n = M._cols;
	int lda = M._rows;

	if ( GPU ) {
		int nb = magma_get_ssytrd_nb(n);
		int ldwa = n;
		float *wA = new float[ldwa * n];
		int lwork = max(2*n + n*nb, 1 + 6*n + 2*n*n);
		float *work = new float[lwork];
		int liwork = 3 + 5*n;
		int *iwork = new int[liwork];
		int info;

		magma_ssyevd_gpu(
			MagmaVec, MagmaUpper,
			n, V._data_gpu, lda,
			D._data_cpu,
			wA, ldwa,
			work, lwork,
			iwork, liwork,
			&info
		);
		throw_on_fail(info == 0);

		delete[] wA;
		delete[] work;
		delete[] iwork;

		V.gpu_read();
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
		throw_on_fail(info == 0);

		delete[] work;
	}

	// take only positive eigenvalues
	int i = 0;
	while ( i < D._cols && D.elem(0, i) < EPSILON ) {
		i++;
	}

	// take only the n1 largest eigenvalues
	i = max(i, D._cols - n1);

	V = V(i, V._cols);
	D = D(i, D._cols).diagonalize();
}

/**
 * Compute the inverse of a square matrix using LU decomposition.
 */
Matrix Matrix::inverse() const
{
	const Matrix& M = *this;

	log(LL_DEBUG, "debug: M^-1 [%d,%d] <- inv(M [%d,%d])",
		M._rows, M._cols, M._rows, M._cols);

	throw_on_fail(is_square(M));

	int m = M._rows;
	int n = M._cols;
	Matrix M_inv = M;
	int lda = M._rows;

	if ( GPU ) {
		int nb = magma_get_sgetri_nb(n);
		int *ipiv = new int[n];
		int lwork = n * nb;
		float *dwork = (float *)gpu_malloc(lwork * sizeof(float));
		int info;

		magma_sgetrf_gpu(m, n, M_inv._data_gpu, lda,
			ipiv, &info);
		throw_on_fail(info >= 0);

		magma_sgetri_gpu(n, M_inv._data_gpu, lda,
			ipiv, dwork, lwork, &info);
		throw_on_fail(info == 0, "Failed to compute inverse");

		delete[] ipiv;
		gpu_free(dwork);

		M_inv.gpu_read();
	}
	else {
		int *ipiv = new int[n];
		int lwork = n;
		float *work = new float[lwork];

		int info = LAPACKE_sgetrf_work(LAPACK_COL_MAJOR,
			m, n, M_inv._data_cpu, lda,
			ipiv);
		throw_on_fail(info >= 0);

		info = LAPACKE_sgetri_work(LAPACK_COL_MAJOR,
			n, M_inv._data_cpu, lda,
			ipiv, work, lwork);
		throw_on_fail(info == 0, "Failed to compute inverse");

		delete[] ipiv;
		delete[] work;
	}

	return M_inv;
}

/**
 * Compute the mean column of a matrix.
 */
Matrix Matrix::mean_column() const
{
	const Matrix& M = *this;

	log(LL_DEBUG, "debug: mu [%d,%d] <- mean(M [%d,%d], 2)",
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

	log(LL_DEBUG, "debug: mu [%d,%d] <- mean(M [%d,%d], 1)",
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
 * Compute the 2-norm of a vector.
 */
float Matrix::norm() const
{
	const Matrix& v = *this;

	log(LL_DEBUG, "debug: n = norm(v [%d,%d])",
		v._rows, v._cols);

	throw_on_fail(is_vector(v));

	int n = length(v);
	int incX = 1;
	float norm;

	if ( GPU ) {
		magma_queue_t queue = magma_queue();

		norm = magma_snrm2(n, v._data_gpu, incX, queue);
	}
	else {
		norm = cblas_snrm2(n, v._data_cpu, incX);
	}

	return norm;
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
	int k1 = A._transposed ? A._rows : A._cols;
	int k2 = B._transposed ? B._cols : B._rows;
	int n = B._transposed ? B._rows : B._cols;

	log(LL_DEBUG, "debug: C [%d,%d] <- A%s [%d,%d] * B%s [%d,%d]",
		m, n,
		A._transposed ? "'" : "", m, k1,
		B._transposed ? "'" : "", k2, n);

	throw_on_fail(k1 == k2);

	Matrix C = Matrix::zeros(m, n);

	float alpha = 1.0f;
	float beta = 0.0f;

	// C := alpha * A * B + beta * C
	if ( GPU ) {
		magma_queue_t queue = magma_queue();
		magma_trans_t TransA = A._transposed ? MagmaTrans : MagmaNoTrans;
		magma_trans_t TransB = B._transposed ? MagmaTrans : MagmaNoTrans;

		magma_sgemm(TransA, TransB,
			m, n, k1,
			alpha, A._data_gpu, A._rows, B._data_gpu, B._rows,
			beta, C._data_gpu, C._rows,
			queue);

		C.gpu_read();
	}
	else {
		CBLAS_TRANSPOSE TransA = A._transposed ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE TransB = B._transposed ? CblasTrans : CblasNoTrans;

		cblas_sgemm(CblasColMajor, TransA, TransB,
			m, n, k1,
			alpha, A._data_cpu, A._rows, B._data_cpu, B._rows,
			beta, C._data_cpu, C._rows);
	}

	return C;
}

/**
 * Compute the sum of the elements of a vector.
 */
float Matrix::sum() const
{
	const Matrix& v = *this;

	log(LL_DEBUG, "debug: s = sum(v [%d,%d])",
		v._rows, v._cols);

	throw_on_fail(is_vector(v));

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

	log(LL_DEBUG, "debug: U, S, V <- svd(M [%d,%d])",
		M._rows, M._cols);

	int m = M._rows;
	int n = M._cols;
	int lda = m;
	int ldu = m;
	int ldvt = min(m, n);

	U = Matrix(ldu, min(m, n));
	S = Matrix(1, min(m, n));
	Matrix VT = Matrix(ldvt, n);

	if ( GPU ) {
		Matrix wA = M;
		int nb = magma_get_sgesvd_nb(m, n);
		int lwork = 2 * min(m, n) + (max(m, n) + min(m, n)) * nb;
		float *work = new float[lwork];
		int info;

		magma_sgesvd(
			MagmaSomeVec, MagmaSomeVec,
			m, n, wA._data_cpu, lda,
			S._data_cpu,
			U._data_cpu, ldu,
			VT._data_cpu, ldvt,
			work, lwork,
			&info);
		throw_on_fail(info == 0);

		delete[] work;
	}
	else {
		Matrix wA = M;
		int lwork = 5 * min(m, n);
		float *work = new float[lwork];

		int info = LAPACKE_sgesvd_work(
			LAPACK_COL_MAJOR, 'S', 'S',
			m, n, wA._data_cpu, lda,
			S._data_cpu,
			U._data_cpu, ldu,
			VT._data_cpu, ldvt,
			work, lwork);
		throw_on_fail(info == 0);

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

	log(LL_DEBUG, "debug: M' [%d,%d] <- transpose(M [%d,%d])",
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

	log(LL_DEBUG, "debug: A [%d,%d] <- A [%d,%d] + B [%d,%d]",
		A._rows, A._cols,
		A._rows, A._cols,
		B._rows, B._cols);

	throw_on_fail(A._rows == B._rows && A._cols == B._cols);

	int n = A._rows * A._cols;
	float alpha = 1.0f;
	int incX = 1;
	int incY = 1;

	if ( GPU ) {
		magma_queue_t queue = magma_queue();

		magma_saxpy(n, alpha, B._data_gpu, incX, A._data_gpu, incY, queue);

		A.gpu_read();
	}
	else {
		cblas_saxpy(n, alpha, B._data_cpu, incX, A._data_cpu, incY);
	}
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

	log(LL_DEBUG, "debug: A(:, %d) [%d,%d] <- B(:, %d) [%d,%d]",
		i + 1, A._rows, 1,
		j + 1, B._rows, 1);

	throw_on_fail(A._rows == B._rows);
	throw_on_fail(0 <= i && i < A._cols);
	throw_on_fail(0 <= j && j < B._cols);

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

	log(LL_DEBUG, "debug: A(%d, :) [%d,%d] <- B(%d, :) [%d,%d]",
		i + 1, 1, A._cols,
		j + 1, 1, B._cols);

	throw_on_fail(A._cols == B._cols);
	throw_on_fail(0 <= i && i < A._rows);
	throw_on_fail(0 <= j && j < B._rows);

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

	log(LL_DEBUG, "debug: M [%d,%d] <- f(M [%d,%d])",
		M._rows, M._cols, M._rows, M._cols);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) = f(M.elem(i, j));
		}
	}

	M.gpu_write();
}

/**
 * Multiply a matrix by a scalar.
 *
 * @param c
 */
void Matrix::elem_mult(float c)
{
	Matrix& M = *this;

	log(LL_DEBUG, "debug: M [%d,%d] <- %g * M [%d,%d]",
		M._rows, M._cols, c, M._rows, M._cols);

	int n = M._rows * M._cols;
	int incX = 1;

	if ( GPU ) {
		magma_queue_t queue = magma_queue();

		magma_sscal(n, c, M._data_gpu, incX, queue);

		M.gpu_read();
	}
	else {
		cblas_sscal(n, c, M._data_cpu, incX);
	}
}

/**
 * Subtract a matrix from another matrix.
 *
 * @param B
 */
void Matrix::subtract(const Matrix& B)
{
	Matrix& A = *this;

	log(LL_DEBUG, "debug: A [%d,%d] <- A [%d,%d] - B [%d,%d]",
		A._rows, A._cols,
		A._rows, A._cols,
		B._rows, B._cols);

	throw_on_fail(A._rows == B._rows && A._cols == B._cols);

	int n = A._rows * A._cols;
	float alpha = -1.0f;
	int incX = 1;
	int incY = 1;

	if ( GPU ) {
		magma_queue_t queue = magma_queue();

		magma_saxpy(n, alpha, B._data_gpu, incX, A._data_gpu, incY, queue);

		A.gpu_read();
	}
	else {
		cblas_saxpy(n, alpha, B._data_cpu, incX, A._data_cpu, incY);
	}
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

	log(LL_DEBUG, "debug: M [%d,%d] <- M [%d,%d] - a [%d,%d] * 1_N' [%d,%d]",
		M._rows, M._cols,
		M._rows, M._cols,
		a._rows, a._cols,
		1, M._cols);

	throw_on_fail(M._rows == a._rows && a._cols == 1);

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

	log(LL_DEBUG, "debug: M [%d,%d] <- M [%d,%d] - a [%d,%d] * 1_N [%d,%d]",
		M._rows, M._cols,
		M._rows, M._cols,
		M._rows, 1,
		a._rows, a._cols);

	throw_on_fail(M._cols == a._cols && a._rows == 1);

	for ( int i = 0; i < M._rows; i++ ) {
		for ( int j = 0; j < M._cols; j++ ) {
			M.elem(i, j) -= a.elem(0, j);
		}
	}
	M.gpu_write();
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
