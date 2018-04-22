/**
 * @file math/matrix.h
 *
 * Interface definitions for the matrix type.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include "mlearn/util/buffer.h"
#include "mlearn/util/iodevice.h"



namespace ML {



typedef float (*elem_func_t)(float);



class Matrix {
private:
	int _rows;
	int _cols;
	Buffer<float> _buffer;
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
	friend IODevice& operator<<(IODevice& file, const Matrix& M);
	friend IODevice& operator>>(IODevice& file, Matrix& M);

	void print() const;
	void gpu_read() { buffer().read(); }
	void gpu_write() { buffer().write(); }

	// getter functions
	int rows() const { return _rows; }
	int cols() const { return _cols; }
	const Buffer<float>& buffer() const { return _transposed ? _T->_buffer : _buffer; }
	Buffer<float>& buffer() { return _transposed ? _T->_buffer : _buffer; }
	const float& elem(int i, int j) const { return buffer().host_data()[j * _rows + i]; }
	float& elem(int i, int j) { return buffer().host_data()[j * _rows + i]; }
	const Matrix& T() const { return *_T; }

	float determinant() const;
	Matrix diagonalize() const;
	void eigen(int n1, Matrix& V, Matrix& D) const;
	Matrix inverse() const;
	Matrix mean_column() const;
	Matrix mean_row() const;
	Matrix product(const Matrix& B) const;
	float sum() const;
	void svd(Matrix& U, Matrix& S, Matrix& V) const;
	Matrix transpose() const;

	// mutator functions
	void add(const Matrix& B);
	void assign_column(int i, const Matrix& B, int j);
	void assign_row(int i, const Matrix& B, int j);
	void elem_apply(elem_func_t f);
	void subtract(const Matrix& B);
	void subtract_columns(const Matrix& a);
	void subtract_rows(const Matrix& a);

	// BLAS wrapper functions
	void axpy(float alpha, const Matrix& A);
	float dot(const Matrix& y) const;
	void gemm(float alpha, const Matrix& A, const Matrix& B, float beta);
	float nrm2() const;
	void scal(float c);
	void syr(float alpha, const Matrix& x);
	void syrk(bool trans, float alpha, const Matrix& A, float beta);

	// LAPACK wrapper functions
	void gesvd(Matrix& U, Matrix& S, Matrix& VT) const;
	void getrf(Matrix& U, Buffer<int>& ipiv) const;
	bool getrs(const Matrix& A, Matrix& B, Buffer<int>& ipiv) const;
	void syev(Matrix& V, Matrix& D) const;

	// operators
	inline Matrix operator()(int i, int j) const { return Matrix(*this, i, j); }
	inline Matrix operator()(int i) const { return (*this)(i, i + 1); }
	inline Matrix& operator=(Matrix B) { swap(*this, B); return *this; }
	inline Matrix& operator+=(const Matrix& B) { add(B); return *this; }
	inline Matrix& operator-=(const Matrix& B) { subtract(B); return *this; }
	inline Matrix& operator*=(float c) { scal(c); return *this; }
	inline Matrix& operator/=(float c) { scal(1 / c); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};



inline Matrix operator+(Matrix A, const Matrix& B) { return (A += B); }
inline Matrix operator-(Matrix A, const Matrix& B) { return (A -= B); }
inline Matrix operator*(const Matrix& A, const Matrix& B) { return A.product(B); }
inline Matrix operator*(Matrix A, float c) { return (A *= c); }
inline Matrix operator*(float c, Matrix A) { return (A *= c); }
inline Matrix operator/(Matrix A, float c) { return (A /= c); }



}

#endif
