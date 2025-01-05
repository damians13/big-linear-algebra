#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Matrix* make_matrix(int rows, int cols, matrix_float_t* data) {
	struct Matrix* p = malloc(sizeof(struct Matrix));
	p->rows = rows;
	p->cols = cols;
	p->data = data;
	return p;
}

struct Matrix* clone_matrix(struct Matrix m) {
	matrix_float_t* data = malloc(m.cols * m.rows * sizeof(matrix_float_t));
	struct Matrix* n = make_matrix(m.rows, m.cols, data);
	for (int i = 0; i < m.cols * m.rows; i++) {
		data[i] = m.data[i];
	}
	return n;
}

// Stack allocated matrix with heap allocated data cleanup
void free_matrix_data(struct Matrix* m) {
	free(m->data);
}

// Heap allocated matrix with heap allocated data cleanup
void free_matrix(struct Matrix* m) {
	free_matrix_data(m);
	free(m);
}

// If a is mxn anad b is nxp, then the resulting matrix is mxp
struct Matrix* matrix_multiply(struct Matrix a, struct Matrix b) {
	if (a.cols != b.rows) {
		printf("Attempted to multiply %dx%d matrix by %dx%d matrix, exiting\n", a.rows, a.cols, b.rows, b.cols);
		exit(1);
	}
	matrix_float_t* data = malloc(b.cols * a.rows * sizeof(matrix_float_t));
	struct Matrix* m = make_matrix(a.rows, b.cols, data);
	for (int i = 0; i < b.cols; i++) {
		for (int j = 0; j < a.rows; j++) {
			matrix_float_t sum = 0;
			for (int k = 0; k < a.cols; k++) {
				sum += a.data[j * a.cols + k] * b.data[k * b.cols + i];
			}
			data[j * b.cols + i] = sum;
		}
	}
	return m;
}

void matrix_scale(struct Matrix* m, matrix_float_t f) {
	for (int i = 0; i < m->cols * m->rows; i++) {
		m->data[i] *= f;
	}
}

void matrix_add(struct Matrix* a, struct Matrix* b) {
	for (int i = 0; i < a->cols * a->rows; i++) {
		a->data[i] += b->data[i];
	}
}

void print_matrix(struct Matrix m) {
	printf("%d x %d matrix\n", m.rows, m.cols);
	for (int i = 0; i < m.rows * m.cols; i++) {
		if (i % m.cols == 0) {
			printf("[ ");
		}
		if (m.data[i] == 0) {
			printf("0 ");
		} else if (m.data[i] < 0.01) {
			printf("%.2e ", m.data[i]);
		}  else {
			printf("%.2f ", m.data[i]);
		}
		if ((i + 1) % m.cols == 0) {
			printf("]\n");
		}
	}
	printf("\n");
}

void print_matrix_dim(struct Matrix m) {
	printf("%d x %d matrix\n", m.rows, m.cols);
}

void matrix_multiply_elementwise(struct Matrix* a, struct Matrix* b) {
	if (a->cols != b->cols || a->rows != b->rows) {
		printf("Attempted to multiply elements of %dx%d matrix by %dx%d matrix, exiting\n", a->rows, a->cols, b->rows, b->cols);
		exit(1);
	}
	for (int i = 0; i < a->cols * a->rows; i++) {
		a->data[i] *= b->data[i];
	}
}

void matrix_transpose(struct Matrix* m) {
	struct Matrix* clone = clone_matrix(*m);

	m->rows = m->cols;
	m->cols = clone->rows;

	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->data[i * m->cols + j] = clone->data[j * clone->cols + i];
		}
	}

	free_matrix(clone);
}

/**
 * Sum the values along the rows (ie. values in the same column)
 */
struct Matrix* matrix_row_sum(struct Matrix m) {
	matrix_float_t* data = malloc(m.cols * sizeof(matrix_float_t));
	struct Matrix* result = make_matrix(1, m.cols, data);
	for (int i = 0; i < m.cols; i++) {
		data[i] = 0;
		for (int j = 0; j < m.rows; j++) {
			data[i] += m.data[i + j * m.cols];
		}
	}
	return result;
}

/**
 * Sum the values along the cols (ie. values in the same row)
 */
struct Matrix* matrix_col_sum(struct Matrix m) {
	matrix_float_t* data = malloc(m.rows * sizeof(matrix_float_t));
	struct Matrix* result = make_matrix(m.rows, 1, data);
	for (int i = 0; i < m.rows; i++) {
		data[i] = 0;
		for (int j = 0; j < m.cols; j++) {
			data[i] += m.data[i * m.rows + j];
		}
	}
	return result;
}

matrix_float_t frobenius_norm(struct Matrix m) {
	matrix_float_t norm_squared = 0;
	for (int i = 0; i < m.cols; i++) {
		for (int j = 0; j < m.rows; j++) {
			norm_squared += m.data[i + j * m.cols] * m.data[i + j * m.cols];
		}
	}
	return sqrt(norm_squared);
}

matrix_float_t max_value(struct Matrix m) {
	matrix_float_t max = -INFINITY;
	for (int i = 0; i < m.cols * m.rows; i++) {
		if (m.data[i] > max) {
			max = m.data[i];
		}
	}
	return max;
}

void matrix_z_score_normalize(Matrix* m) {
	// Calculate mean and standard deviation of the values in m
	matrix_float_t sum = 0;
	matrix_float_t sum_squares = 0;
	for (int i = 0; i < m->cols * m->rows; i++) {
		sum += m->data[i];
		sum_squares += m->data[i] * m->data[i];
	}
	matrix_float_t mean = sum / (matrix_float_t) (m->cols * m->rows);
	matrix_float_t standard_deviation = sqrtf(sum_squares / (matrix_float_t) (m->cols * m->rows) - mean * mean);

	// Normalize the values of m to have mean 0 and standard deviation 1
	for (int i = 0; i < m->cols * m->rows; i++) {
		m->data[i] = (m->data[i] - mean) / standard_deviation;
	}
}

// Tiles the columns of matrix `b` (ie. reuses the columns) and adds the result to `a`.
// Requires that a and b have the same number of rows
void matrix_add_tile_columns(struct Matrix* a, struct Matrix* b) {
	for (int i = 0; i < a->cols; i++) {
		for (int j = 0; j < a->rows; j++) {
			a->data[i + j * a->cols] += b->data[i % b->cols + j * b->cols];
		}
	}
}