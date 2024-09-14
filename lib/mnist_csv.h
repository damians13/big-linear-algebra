#ifndef __mnist_csv_h__
#define __mnist_csv_h__
#include <stdio.h>

// Buffer should have space for 785 ints
struct MnistCSV {
	FILE* file;
	int* buffer;
};

// Populates the integer buffer array where the 1st entry is the numerical value and the remaining 784 are pixel values 0-255. Returns 1 if CSV file is empty, 0 otherwise
int get_next_data(struct MnistCSV* csv);
void visualize_digit_data(struct MnistCSV* csv);

#endif