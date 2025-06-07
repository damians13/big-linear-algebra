#include "util.h"
#include <math.h>
#include <stdlib.h>

void relu(matrix_float_t* data, int num) {
	for (int i = 0; i < num; i++) {
		if (data[i] < 0) {
			data[i] = 0;
		}
	}
}

void softmax(matrix_float_t* data, int rows, int cols) {
	for (int i = 0; i < cols; i++) {
		// Subtract the max value from each value to avoid overflow in expf
		matrix_float_t max_value = -INFINITY;
		for (int j = 0; j < rows; j++) {
			if (data[i + j * cols] > max_value) {
				max_value = data[i + j * cols];
			}
		}

		matrix_float_t sum_of_exponents = 0;
		for (int j = 0; j < rows; j++) {
			data[i + j * cols] = exp(data[i + j * cols] - max_value);
			sum_of_exponents += data[i + j * cols];
		}
		for (int j = 0; j < rows; j++) {
			data[i + j * cols] /= sum_of_exponents;
		}
	}
}

void softmax_row_wise(matrix_float_t* data, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		// Subtract the max value from each value to avoid overflow in expf
		matrix_float_t max_value = -INFINITY;
		for (int j = 0; j < cols; j++) {
			if (data[j + i * cols] > max_value) {
				max_value = data[j + i * cols];
			}
		}

		matrix_float_t sum_of_exponents = 0;
		for (int j = 0; j < cols; j++) {
			data[j + i * cols] = exp(data[j + i * cols] - max_value);
			sum_of_exponents += data[j + i * cols];
		}
		for (int j = 0; j < cols; j++) {
			data[j + i * cols] /= sum_of_exponents;
		}
	}
}

void load_matrix_from_csv(Matrix* m, const char* filepath, int rows, int cols) {
	float* csv_data = read_csv_contents(filepath);
	for (int i = 0; i < rows * cols; i++) {
		m->data[i] = (matrix_float_t) csv_data[i];
	}
	free(csv_data);
	m->rows = rows;
	m->cols = cols;
}