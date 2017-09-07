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

void gpu_init();
void gpu_finalize();

typedef float precision_t;

#define ELEM(M, i, j) (M)._data_cpu[(j) * (M)._rows + (i)]
#define TRAN(M) (*M.T)

typedef precision_t (*elem_func_t)(precision_t);

class Matrix {
private:
	int _rows;
	int _cols;
	precision_t *_data_cpu;
	precision_t *_data_gpu;
	bool _transposed;

public:
	Matrix *T;

	// constructor, destructor functions
	Matrix(int rows, int cols);
	Matrix(int rows, int cols, precision_t *data);
	Matrix(const Matrix& M, int i, int j);
	Matrix(const Matrix& M);
	Matrix(Matrix&& M);
	Matrix();
	~Matrix();

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
	inline precision_t& elem(int i, int j) const { return ELEM(*this, i, j); }

	precision_t determinant() const;
	Matrix diagonalize() const;
	void eigen(int n1, Matrix& V, Matrix& D) const;
	Matrix inverse() const;
	Matrix mean_column() const;
	Matrix mean_row() const;
	precision_t norm() const;
	Matrix product(const Matrix& B) const;
	precision_t sum() const;
	void svd(Matrix& U, Matrix& S, Matrix& V) const;
	Matrix transpose() const;

	// mutator functions
	void add(const Matrix& B);
	void assign_column(int i, const Matrix& B, int j);
	void assign_row(int i, const Matrix& B, int j);
	void elem_apply(elem_func_t f);
	void elem_mult(precision_t c);
	void subtract(const Matrix& B);
	void subtract_columns(const Matrix& a);
	void subtract_rows(const Matrix& a);

	// operators
	inline Matrix operator()(int i, int j) const { return Matrix(*this, i, j); }
	inline Matrix operator()(int i) const { return (*this)(i, i + 1); }
	inline Matrix& operator=(Matrix B) { swap(*this, B); return *this; }
	inline Matrix& operator+=(const Matrix& B) { this->add(B); return *this; }
	inline Matrix& operator-=(const Matrix& B) { this->subtract(B); return *this; }
	inline Matrix& operator*=(precision_t c) { this->elem_mult(c); return *this; }
	inline Matrix& operator/=(precision_t c) { this->elem_mult(1 / c); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};

inline Matrix operator+(Matrix A, const Matrix& B) { return (A += B); }
inline Matrix operator-(Matrix A, const Matrix& B) { return (A -= B); }
inline Matrix operator*(const Matrix& A, const Matrix& B) { return A.product(B); }
inline Matrix operator*(Matrix A, precision_t c) { return (A *= c); }
inline Matrix operator*(precision_t c, Matrix A) { return (A *= c); }
inline Matrix operator/(Matrix A, precision_t c) { return (A /= c); }
inline std::ostream& operator<<(std::ostream& os, const Matrix& M) { M.print(os); return os; }

}

#endif
