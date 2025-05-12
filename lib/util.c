#include "util.h"
#include <math.h>

void relu(matrix_float_t* data, int num) {
	for (int i = 0; i < num; i++) {
		if (data[i] < 0) {
			data[i] = 0;
		}
	}
}

void relu_ddx(matrix_float_t* data, int num) {
	for (int i = 0; i < num; i++) {
		data[i] = data[i] > 0 ? 1 : 0;
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