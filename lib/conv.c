#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// `in` must be an arrays of `Matrix`es of size `in_channels`. Uses "same" padding
void _im2col(Matrix* in, Matrix* out, int kernel_size, int in_channels, int stride) {
    int height = in[0].rows;
    int width = in[0].cols;

    // Formulas from a google search
    int vertical_padding = (ceil(((float) height) / stride) - 1) * stride + kernel_size - height;
    if (vertical_padding < 0) {
        vertical_padding = 0;
    }
    int horizontal_padding = (ceil(((float) width) / stride) - 1) * stride + kernel_size - width;
    if (horizontal_padding < 0) {
        horizontal_padding = 0;
    }
    int pad_top = vertical_padding / 2; // Floor
    int pad_bottom = (vertical_padding + 1) / 2; // Ceil
    int pad_left = horizontal_padding / 2; // Floor
    int pad_right = (horizontal_padding + 1) / 2; // Ceil

    int padded_height = vertical_padding + height;
    int padded_width = horizontal_padding + width;
    int padded_size = padded_width * padded_height;

    matrix_float_t* padded_data = malloc(in_channels * padded_size * sizeof(matrix_float_t));
    for (int c = 0; c < in_channels; c++) {
        for (int i = 0; i < pad_top; i++) {
            for (int j = 0; j < padded_width; j++) {
                padded_data[c * padded_size + i * padded_width + j] = 0;
            }
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < pad_left; j++) {
                padded_data[c * padded_size + (i + pad_top) * padded_width + j] = 0;
            }
            for (int j = 0; j < width; j++) {
                padded_data[c * padded_size + (i + pad_top) * padded_width + j + pad_left] = in[c].data[i * width + j];
            }
            for (int j = 0; j < pad_right; j++) {
                padded_data[c * padded_size + (i + pad_top) * padded_width + j + pad_left + width] = 0;
            }
        }
        for (int i = 0; i < pad_bottom; i++) {
            for (int j = 0; j < padded_width; j++) {
                padded_data[c * padded_size + (i + pad_top + height) * padded_width + j] = 0;
            }
        }
    }

    int out_row_width = kernel_size * kernel_size * in_channels;
    int out_height = (int) ceil((float) height / stride);
    int out_width = (int) ceil((float) width / stride);

    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            int out_row_index = i * out_width + j;
            for (int c = 0; c < in_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        int out_index = out_row_index * out_row_width + c * kernel_size * kernel_size + k * kernel_size + l;
                        int in_row = i * stride + k;
                        int in_col = j * stride + l;
                        int in_index = c * padded_size + in_row * padded_width + in_col;
                        out->data[out_index] = padded_data[in_index];
                    }
                }
            }
        }
    }

    free(padded_data);
}

// `out` and `kernel` must be arrays of `Matrix` of size `out_channels`. Uses "same" padding
void _col2im(Matrix* in, Matrix* out, int kernel_size, int out_channels, int stride) {
    int height = out[0].rows;
    int width = out[0].cols;

    // Formulas from a google search
    int vertical_padding = (ceil(((float) height) / stride) - 1) * stride + kernel_size - height;
    if (vertical_padding < 0) {
        vertical_padding = 0;
    }
    int horizontal_padding = (ceil(((float) width) / stride) - 1) * stride + kernel_size - width;
    if (horizontal_padding < 0) {
        horizontal_padding = 0;
    }
    int pad_top = vertical_padding / 2; // Floor
    int pad_bottom = (vertical_padding + 1) / 2; // Ceil
    int pad_left = horizontal_padding / 2; // Floor
    int pad_right = (horizontal_padding + 1) / 2; // Ceil

    int padded_height = vertical_padding + height;
    int padded_width = horizontal_padding + width;
    int padded_size = padded_width * padded_height;

    matrix_float_t* padded_data = malloc(out_channels * padded_size * sizeof(matrix_float_t));

    int in_row_width = kernel_size * kernel_size * out_channels;

    memset(padded_data, 0, out_channels * padded_size * sizeof(matrix_float_t));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int in_row_index = i * width + j;
            for (int c = 0; c < out_channels; c++) {
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        int in_index = in_row_index * in_row_width + c * kernel_size * kernel_size + k * kernel_size + l;
                        int out_row = (i * stride) + k;
                        int out_col = (j * stride) + l;
                        int out_index = c * padded_size + out_row * padded_width + out_col;
                        padded_data[out_index] += in->data[in_index];
                    }
                }
            }
        }
    }

    // Crop out padding
    for (int c = 0; c < out_channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                out[c].data[i * width + j] = padded_data[c * padded_size + (i + pad_top) * padded_width + j + pad_left];
            }
        }
    }

    free(padded_data);
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
void reshape_channels_matrix(Matrix* channels, Matrix* matrix) {
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
void reshape_matrix_channels(Matrix* matrix, Matrix* channels) {
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
    reshape_matrix_channels(data->product, data->output); // (output_channels, output_height, output_width)
}

void conv_ddx(Matrix* del_Y, ConvData* data, ConvData* grad_data, Matrix** del_kernels, Matrix* del_input, int in_channels, int stride) {
    Matrix* del_Q = grad_data->product;
    Matrix* del_kernels_matrix = grad_data->kernel_matrix;
    Matrix* del_input_matrix = grad_data->im2col;
    int kernel_size = del_kernels[0][0].cols;

    reshape_channels_matrix(del_Y, del_Q);
    matrix_transpose(data->im2col);
    matrix_multiply_inplace(data->im2col, del_Q, del_kernels_matrix);
    _reshape_matrix_kernels(del_kernels_matrix, del_kernels);
    matrix_transpose(data->im2col);
    matrix_transpose(data->kernel_matrix);
    matrix_multiply_inplace(del_Q, data->kernel_matrix, del_input_matrix);
    matrix_transpose(data->kernel_matrix);
    _col2im(del_input_matrix, del_input, kernel_size, in_channels, stride);
}