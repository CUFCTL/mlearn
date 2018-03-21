/**
 * @file test_matrix.cpp
 *
 * Test suite for the matrix library.
 *
 * Tests are based on examples in the MATLAB documentation
 * where appropriate.
 */
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <mlearn.h>



using namespace ML;



#define ANSI_RED    "\x1b[31m"
#define ANSI_BOLD   "\x1b[1m"
#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RESET  "\x1b[0m"



typedef void (*test_func_t)(void);



/**
 * Determine whether two floating point values are equal.
 *
 * @param a
 * @param b
 */
bool is_equal(float a, float b)
{
	static float EPSILON = 1e-4;

	return (fabsf(a - b) < EPSILON);
}



/**
 * Determine whether two matrices are equal.
 *
 * @param A
 * @param B
 */
bool m_equal(const Matrix& A, const Matrix& B)
{
	if ( A.rows() != B.rows() || A.cols() != B.cols() ) {
		return false;
	}

	for ( int i = 0; i < A.rows(); i++ ) {
		for ( int j = 0; j < A.cols(); j++ ) {
			if ( !is_equal(A.elem(i, j), B.elem(i, j)) ) {
				return false;
			}
		}
	}

	return true;
}



/**
 * Print a test result.
 *
 * @param name
 * @param result
 */
void print_result(const char *name, bool result)
{
	std::string color = result ? ANSI_GREEN : ANSI_RED;
	std::string message = result ? "PASSED" : "FAILED";

	std::cout << color << std::left << std::setw(25) << name << "  " << message << ANSI_RESET << "\n";
}



/**
 * Assert that two floating-point values are equal.
 *
 * @param a
 * @param b
 * @param name
 */
void assert_equal(float a, float b, const char *name)
{
	print_result(name, is_equal(a, b));
}



/**
 * Assert that two matrices are equal.
 *
 * @param A
 * @param B
 * @param name
 */
void assert_equal_matrix(const Matrix& A, const Matrix& B, const char *name)
{
	print_result(name, m_equal(A, B));
}



/**
 * Assert that a matrix M is equal to a test value.
 *
 * @param M
 * @param data
 * @param name
 */
void assert_matrix_value(const Matrix& M, float *data, const char *name)
{
	Matrix M_test(M.rows(), M.cols(), data);

	assert_equal_matrix(M, M_test, name);
}



/**
 * Test identity matrix.
 */
void test_identity()
{
	float I_data[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	Matrix I = Matrix::identity(4);

	if ( Logger::test(LogLevel::Verbose) ) {
		I.print();
	}

	assert_matrix_value(I, I_data, "eye(N)");
}



/**
 * Test ones matrix.
 */
void test_ones()
{
	float X_data[] = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1
	};
	Matrix X = Matrix::ones(4, 4);

	if ( Logger::test(LogLevel::Verbose) ) {
		X.print();
	}

	assert_matrix_value(X, X_data, "ones(M, N)");
}



/**
 * Test zero matrix.
 */
void test_zeros()
{
	float X_data[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	Matrix X = Matrix::zeros(4, 4);

	if ( Logger::test(LogLevel::Verbose) ) {
		X.print();
	}

	assert_matrix_value(X, X_data, "zeros(M, N)");
}



/**
 * Test matrix copy.
 */
void test_copy()
{
	float A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	Matrix A(4, 4, A_data);
	Matrix C(A);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		C.print();
	}

	assert_equal_matrix(A, C, "A(:, :)");
}



/**
 * Test matrix column copy.
 */
void test_copy_columns()
{
	float A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	float C_data[] = {
		 2,  3,
		11, 10,
		 7,  6,
		14, 15
	};
	Matrix A(4, 4, A_data);

	int i = 1;
	int j = 3;
	Matrix C = A(i, j);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		C.print();
	}

	assert_matrix_value(C, C_data, "A(:, i:j)");
}



/**
 * The the matrix determinant.
 */
void test_determinant()
{
	float A_data[] = {
		 2, -1,  0,
		-1,  2, -1,
		 0, -1,  2
	};
	Matrix A(3, 3, A_data);

	float det = A.determinant();

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		std::cout << "det(A) = " << det << "\n";
	}

	assert_equal(det, 4, "det(A)");
}



/**
 * Test the diagonal matrix.
 */
void test_diagonalize()
{
	float v_data[] = {
		2, 1, -1, -2, -5
	};
	float D_data[] = {
		2,  0,  0,  0,  0,
		0,  1,  0,  0,  0,
		0,  0, -1,  0,  0,
		0,  0,  0, -2,  0,
		0,  0,  0,  0, -5
	};
	Matrix v(1, 5, v_data);
	Matrix D = v.diagonalize();

	if ( Logger::test(LogLevel::Verbose) ) {
		v.print();
		D.print();
	}

	assert_matrix_value(D, D_data, "diag(v)");
}



/**
 * Test dot product.
 */
void test_dot()
{
	float a_data[] = {
		1, 1, 0, 0
	};
	float b_data[] = {
		1, 2, 3, 4
	};
	Matrix a(1, 4, a_data);
	Matrix b(1, 4, b_data);
	float d = a.dot(b);

	if ( Logger::test(LogLevel::Verbose) ) {
		a.print();
		b.print();
		std::cout << "a' * b = " << d << "\n";
	}

	assert_equal(d, 3, "a' * b");
}



/**
 * Test eigenvalues, eigenvectors.
 */
void test_eigen()
{
	float M_data[] = {
		1.0000, 0.5000, 0.3333, 0.2500,
		0.5000, 1.0000, 0.6667, 0.5000,
		0.3333, 0.6667, 1.0000, 0.7500,
		0.2500, 0.5000, 0.7500, 1.0000
	};
	float V_data[] = {
		 0.0694, -0.4422, -0.8105,  0.3778,
		-0.3619,  0.7420, -0.1877,  0.5322,
		 0.7694,  0.0487,  0.3010,  0.5614,
		-0.5218, -0.5015,  0.4661,  0.5088
	};
	float D_data[] = {
		0.2078, 0.0000, 0.0000, 0.0000,
		0.0000, 0.4078, 0.0000, 0.0000,
		0.0000, 0.0000, 0.8482, 0.0000,
		0.0000, 0.0000, 0.0000, 2.5362
	};
	Matrix M(4, 4, M_data);
	Matrix V;
	Matrix D;

	M.eigen(M.rows(), V, D);

	if ( Logger::test(LogLevel::Verbose) ) {
		M.print();
		V.print();
		D.print();
	}

	assert_matrix_value(V, V_data, "eigenvectors of M");
	assert_matrix_value(D, D_data, "eigenvalues of M");
}



/**
 * Test matrix inverse.
 */
void test_inverse()
{
	float X_data[] = {
		 2, -1,  0,
		-1,  2, -1,
		 0, -1,  2
	};
	float Y_data[] = {
		0.7500, 0.5000, 0.2500,
		0.5000, 1.0000, 0.5000,
		0.2500, 0.5000, 0.7500
	};
	Matrix X(3, 3, X_data);
	Matrix Y = X.inverse();

	if ( Logger::test(LogLevel::Verbose) ) {
		X.print();
		Y.print();
	}

	assert_matrix_value(Y, Y_data, "inv(X)");
}



/**
 * Test matrix mean column.
 */
void test_mean_column()
{
	float A_data[] = {
		0, 1, 1,
		2, 3, 2
	};
	float m_data[] = {
		0.6667,
		2.3333
	};
	Matrix A(2, 3, A_data);
	Matrix m = A.mean_column();

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		m.print();
	}

	assert_matrix_value(m, m_data, "mean(A, 2)");
}



/**
 * Test matrix mean row.
 */
void test_mean_row()
{
	float A_data[] = {
		0, 1, 1,
		2, 3, 2,
		1, 3, 2,
		4, 2, 2
	};
	float m_data[] = {
		1.7500, 2.2500, 1.7500
	};
	Matrix A(4, 3, A_data);
	Matrix m = A.mean_row();

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		m.print();
	}

	assert_matrix_value(m, m_data, "mean(A, 1)");
}



/**
 * Test vector norm.
 */
void test_nrm2()
{
	float v_data[] = {
		-2, 3, 1
	};
	Matrix v(1, 3, v_data);
	float n = v.nrm2();

	if ( Logger::test(LogLevel::Verbose) ) {
		v.print();
		std::cout << "nrm2(v) = " << n << "\n";
	}

	assert_equal(n, 3.7417, "nrm2(v)");
}



/**
 * Test matrix product.
 */
void test_product()
{
	// multiply two vectors, A * B
	// multiply two vectors, B * A
	float A1_data[] = {
		1, 1, 0, 0
	};
	float B1_data[] = {
		1,
		2,
		3,
		4
	};
	float C1_data[] = {
		3
	};
	float C2_data[] = {
		1, 1, 0, 0,
		2, 2, 0, 0,
		3, 3, 0, 0,
		4, 4, 0, 0
	};
	Matrix A1(1, 4, A1_data);
	Matrix B1(4, 1, B1_data);
	Matrix C1 = A1 * B1;
	Matrix C2 = B1 * A1;

	if ( Logger::test(LogLevel::Verbose) ) {
		A1.print();
		B1.print();
		C1.print();
		C2.print();
	}

	assert_matrix_value(C1, C1_data, "A1 * B1");
	assert_matrix_value(C2, C2_data, "B1 * A1");

	// multiply two matrices
	float A2_data[] = {
		1, 3, 5,
		2, 4, 7
	};
	float B2_data[] = {
		-5, 8, 11,
		 3, 9, 21,
		 4, 0,  8
	};
	float C3_data[] = {
		24, 35, 114,
		30, 52, 162
	};
	Matrix A2(2, 3, A2_data);
	Matrix B2(3, 3, B2_data);
	Matrix C3 = A2 * B2;

	if ( Logger::test(LogLevel::Verbose) ) {
		A2.print();
		B2.print();
		C3.print();
	}

	assert_matrix_value(C3, C3_data, "A2 * B2");
}



/**
 * Test vector sum.
 */
void test_sum()
{
	float v_data[] = {
		-2, 3, 1
	};
	Matrix v(1, 3, v_data);
	float s = v.sum();

	if ( Logger::test(LogLevel::Verbose) ) {
		v.print();

		std::cout << "sum(v) = " << s << "\n";
	}

	assert_equal(s, 2, "sum(v)");
}



/**
 * Test singular value decomposition.
 */
void test_svd()
{
	float A_data[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8
	};
	float U_data[] = {
		-0.1525, -0.8226,
		-0.3499, -0.4214,
		-0.5474, -0.0201,
		-0.7448,  0.3812,
	};
	float S_data[] = {
		14.2691,      0,
		      0, 0.6268
	};
	float V_data[] = {
		-0.6414,  0.7672,
		-0.7672, -0.6414
	};
	Matrix A(4, 2, A_data);
	Matrix U, S, V;

	A.svd(U, S, V);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		U.print();
		S.print();
		V.print();
	}

	assert_matrix_value(U, U_data, "l. singular vectors of A");
	assert_matrix_value(S, S_data, "singular values of A");
	assert_matrix_value(V, V_data, "r. singular vectors of A");
}



/**
 * Test matrix transpose.
 */
void test_transpose()
{
	float A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	float B_data[] = {
		16,  5,  9,  4,
		 2, 11,  7, 14,
		 3, 10,  6, 15,
		13,  8, 12,  1
	};
	Matrix A(4, 4, A_data);
	Matrix B = A.transpose();

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		B.print();
	}

	assert_matrix_value(B, B_data, "A'");
}



/**
 * Test matrix addition.
 */
void test_add()
{
	float A_data1[] = {
		1, 0,
		2, 4
	};
	float A_data2[] = {
		6, 9,
		4, 5
	};
	float B_data[] = {
		5, 9,
		2, 1
	};
	Matrix A(2, 2, A_data1);
	Matrix B(2, 2, B_data);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		B.print();
	}

	A += B;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "A + B");
}



/**
 * Test matrix column assingment.
 */
void test_assign_column()
{
	float A_data1[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	float A_data2[] = {
		16,  2,  0, 13,
		 5, 11,  0,  8,
		 9,  7,  0, 12,
		 4, 14,  0,  1
	};
	float B_data[] = {
		0,
		0,
		0,
		0
	};
	Matrix A(4, 4, A_data1);
	Matrix B(4, 1, B_data);
	int i = 2;
	int j = 0;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		B.print();
	}

	A.assign_column(i, B, j);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "A(:, i) = B(:, j)");
}



/**
 * Test matrix row assingment.
 */
void test_assign_row()
{
	float A_data1[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	float A_data2[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 0,  0,  0,  0,
		 4, 14, 15,  1
	};
	float B_data[] = {
		0, 0, 0, 0
	};
	Matrix A(4, 4, A_data1);
	Matrix B(1, 4, B_data);
	int i = 2;
	int j = 0;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		B.print();
	}

	A.assign_row(i, B, j);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "A(i, :) = B(j, :)");
}



/**
 * Test matrix element-wise function application.
 */
void test_elem_apply()
{
	float A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	float A_data2[] = {
		1.0000, 0.0000, 1.4142,
		1.7321, 1.0000, 2.0000
	};
	Matrix A(2, 3, A_data1);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	A.elem_apply(sqrtf);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "sqrt(A)");
}



/**
 * Test matrix multiplication by scalar.
 */
void test_scal()
{
	float A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	float A_data2[] = {
		3, 0, 6,
		9, 3, 12
	};
	Matrix A(2, 3, A_data1);
	float c = 3;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	A *= c;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "c * A");
}



/**
 * Test matrix subtraction.
 */
void test_subtract()
{
	float A_data1[] = {
		1, 0,
		2, 4
	};
	float A_data2[] = {
		-4, -9,
		 0,  3
	};
	float B_data[] = {
		5, 9,
		2, 1
	};
	Matrix A(2, 2, A_data1);
	Matrix B(2, 2, B_data);

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
		B.print();
	}

	A -= B;

	if ( Logger::test(LogLevel::Verbose) ) {
		A.print();
	}

	assert_matrix_value(A, A_data2, "A - B");
}



/**
 * Test matrix column subtraction.
 */
void test_subtract_columns()
{
	float M_data1[] = {
		0, 2, 1, 4,
		1, 3, 3, 2,
		1, 2, 2, 2
	};
	float M_data2[] = {
		0, 2, 1, 4,
		0, 2, 2, 1,
		0, 1, 1, 1
	};
	float a_data[] = {
		0,
		1,
		1
	};
	Matrix M(3, 4, M_data1);
	Matrix a(3, 1, a_data);;

	if ( Logger::test(LogLevel::Verbose) ) {
		M.print();
		a.print();
	}

	M.subtract_columns(a);

	if ( Logger::test(LogLevel::Verbose) ) {
		M.print();
	}

	assert_matrix_value(M, M_data2, "M - a * 1_N'");
}



/**
 * Test matrix row subtraction.
 */
void test_subtract_rows()
{
	float M_data1[] = {
		0, 2, 1, 4,
		1, 3, 3, 2,
		1, 2, 2, 2
	};
	float M_data2[] = {
		0,  0,  0,  0,
		1,  1,  2, -2,
		1,  0,  1, -2
	};
	float a_data[] = {
		0, 2, 1, 4
	};
	Matrix M(3, 4, M_data1);
	Matrix a(1, 4, a_data);

	if ( Logger::test(LogLevel::Verbose) ) {
		M.print();
		a.print();
	}

	M.subtract_rows(a);

	if ( Logger::test(LogLevel::Verbose) ) {
		M.print();
	}

	assert_matrix_value(M, M_data2, "M - 1_N * a");
}



void print_usage()
{
	std::cerr <<
		"Usage: ./test-matrix [options]\n"
		"\n"
		"Options:\n"
		"  --gpu             use GPU acceleration\n"
		"  --loglevel LEVEL  log level (0=error, 1=warn, [2]=info, 3=verbose, 4=debug)\n";
}



int main(int argc, char **argv)
{
	// parse command-line arguments
	struct option long_options[] = {
		{ "gpu", no_argument, 0, 'g' },
		{ "loglevel", required_argument, 0, 'e' },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, nullptr)) != -1 ) {
		switch ( opt ) {
		case 'g':
			GPU = true;
			break;
		case 'e':
			Logger::LEVEL = (LogLevel) atoi(optarg);
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	gpu_init();

	// run tests
	test_func_t tests[] = {
		test_identity,
		test_ones,
		test_zeros,
		test_copy,
		test_copy_columns,
		test_determinant,
		test_diagonalize,
		test_dot,
		test_eigen,
		test_inverse,
		test_mean_column,
		test_mean_row,
		test_nrm2,
		test_product,
		test_sum,
		test_svd,
		test_transpose,
		test_add,
		test_subtract,
		test_assign_column,
		test_assign_row,
		test_elem_apply,
		test_scal,
		test_subtract_columns,
		test_subtract_rows
	};
	int num_tests = sizeof(tests) / sizeof(test_func_t);

	for ( int i = 0; i < num_tests; i++ ) {
		test_func_t test = tests[i];

		std::cout << "TEST " << i + 1 << "\n";
		test();
		std::cout << "\n";
	}

	gpu_finalize();

	return 0;
}
