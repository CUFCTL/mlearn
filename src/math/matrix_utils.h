/**
 * @file math/matrix_utils.h
 *
 * Library of helpful matrix functions.
 */
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include "data/dataset.h"
#include "math/matrix.h"

namespace ML {

typedef float (*dist_func_t)(const Matrix&, int, const Matrix&, int);

float m_dist_COS(const Matrix& A, int i, const Matrix& B, int j);
float m_dist_L1(const Matrix& A, int i, const Matrix& B, int j);
float m_dist_L2(const Matrix& A, int i, const Matrix& B, int j);

Matrix m_mean(const std::vector<Matrix>& X);

std::vector<Matrix> m_copy_columns(const Matrix& X);
std::vector<Matrix> m_random_sample(const std::vector<Matrix>& X, int k);
std::vector<Matrix> m_subtract_mean(const std::vector<Matrix>& X, const Matrix& mu);

std::vector<Matrix> m_copy_classes(const Matrix& X, const std::vector<DataEntry>& y, int c);
std::vector<Matrix> m_class_means(const std::vector<Matrix>& X_c);
std::vector<Matrix> m_class_scatters(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U);
Matrix m_scatter_between(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U);
Matrix m_scatter_within(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U);

}

#endif
