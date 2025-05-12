#ifndef __util_h__
#define __util_h__

#include "matrix.h"

void relu(matrix_float_t* data, int num);
void relu_ddx(matrix_float_t* data, int num);
void softmax(matrix_float_t* data, int rows, int cols);

#endif