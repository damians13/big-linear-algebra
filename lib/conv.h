#ifndef __conv_h__
#define __conv_h__

#include "matrix.h"

typedef struct ConvData {
    Matrix* im2col;
	Matrix* kernel_matrix;
	Matrix* product;
	Matrix* output;
} ConvData;

void conv(Matrix* X, Matrix** kernels, ConvData* data, int in_channels, int out_channels, int stride);
void reshape_channels_matrix(Matrix* channels, Matrix* matrix);
void reshape_matrix_channels(Matrix* matrix, Matrix* channels);
void conv_ddx(Matrix* del_Y, ConvData* data, ConvData* grad_data, Matrix** del_kernels, Matrix* del_input, int in_channels, int stride);

#endif