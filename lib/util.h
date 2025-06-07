#ifndef __util_h__
#define __util_h__

#include "matrix.h"
#include "csv.h"

void relu(matrix_float_t* data, int num);
void softmax(matrix_float_t* data, int rows, int cols);
void softmax_row_wise(matrix_float_t* data, int rows, int cols);
void load_matrix_from_csv(Matrix* m, const char* filepath, int rows, int cols);
double random_gaussian(unsigned int* seed);

#endif