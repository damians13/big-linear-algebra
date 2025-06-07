#include "util.h"
#include <math.h>
#include <stdlib.h>

const double PI = 3.14159265358979323846;

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

// Sample a random value from a Gaussian distribution with mean 0 and stdev 1. Seed pointer should be thread safe
double random_gaussian(unsigned int* seed) {
	(void) seed;

	double epsilon = 1e-8;
	static double Z1;
	static int available = 0;

	if (available == 0) {
		// Uses the Box-Muller transform to convert uniform random (from rand()) to Gaussian random
		double U1 = (double) rand() / RAND_MAX;
		while (U1 == 0) {
			U1 = (double) rand() / RAND_MAX; // Resample to avoid log(0)
		}
		double U2 = (double) rand() / RAND_MAX;

		double R = sqrt(-2 * log(U1));
		double theta = 2 * PI * U2;
		
		double Z0 = R * cos(theta);
		Z1 = R * sin(theta);

		available = 1;
		return Z0;
	} else {
		available = 0;
		return Z1;
	}
}