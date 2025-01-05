#ifndef __mnist_csv_h__
#define __mnist_csv_h__
#include <stdio.h>

// Buffer should have space for 785 floats
typedef struct MnistCSV {
	FILE* file;
	float* buffer;
	int num_lines;
} MnistCSV;

// Populates the integer buffer array with a single row of data where the 1st entry is the numerical value and the remaining 784 are pixel values 0-255. Returns 1 if CSV file is empty, 0 otherwise
int get_next_data(struct MnistCSV* csv);
void visualize_digit_data(struct MnistCSV* csv);

#endif