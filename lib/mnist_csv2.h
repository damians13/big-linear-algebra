#ifndef __mnist_csv_h__
#define __mnist_csv_h__
#include <stdio.h>

typedef struct MnistCSV {
	FILE* file;
	float* X; // Data
	float* y; // Labels
	int num_examples;
	int num_sampled;
	char* sampled;
} MnistCSV;

typedef struct MnistExample {
	float* X;
	float y;
	int num_examples;
} MnistExample;

void mnist_csv_init(MnistCSV* csv);
// Sample a random data point from a uniform distribution with replacement
MnistExample get_random_data_replace(MnistCSV* csv);
// Sample a random data point from a uniform distribution without replacement
MnistExample get_random_data_take(MnistCSV* csv);
void visualize_digit_data(MnistExample ex);

#endif