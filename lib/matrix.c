#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

struct Matrix* make_matrix(int rows, int cols, float* data) {
	struct Matrix* p = malloc(sizeof(struct Matrix));
	p->rows = rows;
	p->cols = cols;
	p->data = data;
	return p;
}

struct Matrix* clone_matrix(struct Matrix m) {
	float* data = malloc(m.cols * m.rows * sizeof(float));
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

struct Matrix* matrix_multiply(struct Matrix a, struct Matrix b) {
	float* data = malloc(b.cols * a.rows * sizeof(float));
	struct Matrix* m = make_matrix(a.rows, b.cols, data);
	for (int i = 0; i < b.cols; i++) {
		for (int j = 0; j < a.rows; j++) {
			float sum = 0;
			for (int k = 0; k < a.cols; k++) {
				sum += a.data[j * a.cols + k] * b.data[k * b.cols + i];
			}
			data[j * b.cols + i] = sum;
		}
	}
	return m;
}

void matrix_scale(struct Matrix* m, float f) {
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
		printf("%.2f ", m.data[i]);
		if ((i + 1) % m.cols == 0) {
			printf("]\n");
		}
	}
	printf("\n");
}