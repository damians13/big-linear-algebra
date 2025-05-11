#include "conv.h"
#include <stdio.h>
#include <math.h>

// `in` must be an arrays of `Matrix`es of size `in_channels`
void _im2col(Matrix* in, Matrix* out, int kernel_size, int in_channels, int stride) {
    int num_horizontal_convolutions = ceil((float) (in[0].cols - kernel_size) / stride) + 1;
    int num_vertical_convolutions = ceil((float) (in[0].rows - kernel_size) / stride) + 1;
    int out_row_width = kernel_size * kernel_size * in_channels;

    for (int i = 0; i < num_vertical_convolutions; i++) {
        for (int j = 0; j < num_horizontal_convolutions; j++) {
            int out_row_index = i * num_horizontal_convolutions + j;
            for (int c = 0; c < in_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        int out_index = out_row_index * out_row_width + c * kernel_size * kernel_size + k * kernel_size + l;
                        int in_row = i * stride + k;
                        int in_col = j * stride + l;
                        if (in_row < in[c].rows && in_col < in[c].cols) {
                            out->data[out_index] = in[c].data[in_row * in[c].cols + in_col];
                        } else {
                            out->data[out_index] = 0;
                        }
                    }
                }
            }
        }
    }
}

// `out` and `kernel` must be arrays of `Matrix` of size `out_channels`
void _col2im(Matrix* in, Matrix* out, int kernel_size, int out_channels, int stride) {
    int num_horizontal_convolutions = (int)ceil((float)(out[0].cols - kernel_size) / stride) + 1;
    int num_vertical_convolutions = (int)ceil((float)(out[0].rows - kernel_size) / stride) + 1;
    int in_row_width = kernel_size * kernel_size * out_channels;

    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < out[i].rows; j++) {
            for (int k = 0; k < out[i].cols; k++) {
                out[i].data[j * out[i].cols + k] = 0;
            }
        }
    }

    for (int i = 0; i < num_vertical_convolutions; i++) {
        for (int j = 0; j < num_horizontal_convolutions; j++) {
            int in_row_index = i * num_horizontal_convolutions + j;
            for (int c = 0; c < out_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        int in_index = in_row_index * in_row_width + c * kernel_size * kernel_size + k * kernel_size + l;
                        int out_row = (i * stride) + k;
                        int out_col = (j * stride) + l;
                        if (out_row < out[c].rows && out_col < out[c].cols) {
                            out[c].data[out_row * out[c].cols + out_col] += in->data[in_index];
                        }
                    }
                }
            }
        }
    }
}

// Combine kernel height, width, and channel dimensions to produce a matrix. Kernels: (F, C, H, W). Matrix: (H * W * C, F)
void _reshape_kernels_matrix(Matrix** kernels, Matrix* matrix) {
    int kernel_size = kernels[0][0].rows;
    int num_kernels = matrix->cols;
    int num_kernel_channels = matrix->rows / (kernel_size * kernel_size);

    for (int f = 0; f < num_kernels; f++) {
        for (int c = 0; c < num_kernel_channels; c++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int matrix_row = c * kernel_size * kernel_size + i * kernel_size + j;
                    matrix->data[matrix_row * num_kernels + f] = kernels[f][c].data[i * kernel_size + j];
                }
            }
        }
    }
}

// Split up a matrix into kernel height, width, and channel dimensions by kernel. Matrix: (H * W * C, F). Kernels: (F, C, H, W)
void _reshape_matrix_kernels(Matrix* matrix, Matrix** kernels) {
    int kernel_size = kernels[0][0].rows;
    int num_kernels = matrix->cols;
    int num_kernel_channels = matrix->rows / (kernel_size * kernel_size);

    for (int f = 0; f < num_kernels; f++) {
        for (int c = 0; c < num_kernel_channels; c++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int matrix_row = c * kernel_size * kernel_size + i * kernel_size + j;
                    kernels[f][c].data[i * kernel_size + j] = matrix->data[matrix_row * num_kernels + f];
                }
            }
        }
    }
}

// Combine height and width dimensions to produce a matrix. Channels: (C, H, W). Matrix: (H * W, C)
void _reshape_channels_matrix(Matrix* channels, Matrix* matrix) {
    int num_channels = matrix->cols;
    int height = channels[0].rows;
    int width = channels[0].cols;

    for (int c = 0; c < num_channels; c++) { 
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                channels[c].data[index] = matrix->data[index * num_channels + c];
            }
        }
    }
}

// Split up a matrix into height and width dimensions by channel. Channels: (C, H, W). Matrix: (H * W, C)
void _reshape_matrix_channels(Matrix* matrix, Matrix* channels) {
    int num_channels = matrix->cols;
    int height = channels[0].rows;
    int width = channels[0].cols;

    for (int c = 0; c < num_channels; c++) { 
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                matrix->data[index * num_channels + c] = channels[c].data[index];
            }
        }
    }
}

void conv(Matrix* X, Matrix** kernels, ConvData* data, int in_channels, int out_channels, int stride) {
    // X is (image_channels, image_height, image_width) and kernels is (output_channels, image_channels, kernel_height, kernel_width)
    int kernel_size = kernels[0][0].cols;
    _im2col(X, data->im2col, kernel_size, in_channels, stride); // (output_height * output_width, kernel_height * kernel_width * image_channels)
    _reshape_kernels_matrix(kernels, data->kernel_matrix); // (kernel_height * kernel_width * image_channels, output_channels)
    matrix_multiply_inplace(data->im2col, data->kernel_matrix, data->product); // (output_height * output_width, output_channels)
    _reshape_matrix_channels(data->product, data->output); // (output_channels, output_height, output_width)
}