/**
 * @file math/matrix.h
 *
 * Interface definitions for the matrix type.
 *
 * NOTE: Unlike C, which stores static arrays in row-major
 * order, this library stores matrices in column-major order.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <fstream>
#include <iostream>

namespace ML {

typedef float (*elem_func_t)(float);

extern bool GPU;
extern int GPU_DEVICE;

void gpu_init();
void gpu_finalize();

#define ELEM(M, i, j) (M)._data_cpu[(j) * (M)._rows + (i)]

class Matrix {
private:
	int _rows;
	int _cols;
	float *_data_cpu;
	float *_data_gpu;
	bool _transposed;
	Matrix *_T;

public:
	// constructor, destructor functions
	Matrix(int rows, int cols);
	Matrix(int rows, int cols, float *data);
	Matrix(const Matrix& M, int i, int j);
	Matrix(const Matrix& M);
	Matrix(Matrix&& M);
	Matrix();
	~Matrix();

	void init_identity();
	void init_ones();
	void init_random();
	void init_zeros();

	static Matrix identity(int rows);
	static Matrix ones(int rows, int cols);
	static Matrix random(int rows, int cols);
	static Matrix zeros(int rows, int cols);

	// I/O functions
	void print(std::ostream& os) const;

	void save(std::ofstream& file) const;
	void load(std::ifstream& file);

	void gpu_read();
	void gpu_write();

	// getter functions
	inline int rows() const { return this->_rows; }
	inline int cols() const { return this->_cols; }
	inline float& elem(int i, int j) const { return ELEM(*this, i, j); }
	inline Matrix& T() const { return *(this->_T); }

	float determinant() const;
	Matrix diagonalize() const;
	float dot(const Matrix& b) const;
	void eigen(int n1, Matrix& V, Matrix& D) const;
	Matrix inverse() const;
	Matrix mean_column() const;
	Matrix mean_row() const;
	float norm() const;
	Matrix product(const Matrix& B) const;
	float sum() const;
	void svd(Matrix& U, Matrix& S, Matrix& V) const;
	Matrix transpose() const;

	// mutator functions
	void add(const Matrix& B);
	void assign_column(int i, const Matrix& B, int j);
	void assign_row(int i, const Matrix& B, int j);
	void elem_apply(elem_func_t f);
	void elem_mult(float c);
	void subtract(const Matrix& B);
	void subtract_columns(const Matrix& a);
	void subtract_rows(const Matrix& a);

	// BLAS wrapper functions
	void axpy(float alpha, const Matrix& A);
	void gemm(float alpha, const Matrix& A, const Matrix& B, float beta);

	// operators
	inline Matrix operator()(int i, int j) const { return Matrix(*this, i, j); }
	inline Matrix operator()(int i) const { return (*this)(i, i + 1); }
	inline Matrix& operator=(Matrix B) { swap(*this, B); return *this; }
	inline Matrix& operator+=(const Matrix& B) { this->add(B); return *this; }
	inline Matrix& operator-=(const Matrix& B) { this->subtract(B); return *this; }
	inline Matrix& operator*=(float c) { this->elem_mult(c); return *this; }
	inline Matrix& operator/=(float c) { this->elem_mult(1 / c); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};

inline Matrix operator+(Matrix A, const Matrix& B) { return (A += B); }
inline Matrix operator-(Matrix A, const Matrix& B) { return (A -= B); }
inline Matrix operator*(const Matrix& A, const Matrix& B) { return A.product(B); }
inline Matrix operator*(Matrix A, float c) { return (A *= c); }
inline Matrix operator*(float c, Matrix A) { return (A *= c); }
inline Matrix operator/(Matrix A, float c) { return (A /= c); }
inline std::ostream& operator<<(std::ostream& os, const Matrix& M) { M.print(os); return os; }

}

#endif
