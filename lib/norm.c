#include "norm.h"

void group_norm(Matrix* in, Matrix* out, matrix_float_t* stdevs, int channels, int group_size) {
    int num_groups = (channels + group_size - 1) / group_size;
    for (int g = 0; g < num_groups; g++) {
        matrix_float_t mean = 0;
        for (int c = 0; c < group_size; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    mean += in[channel_index].data[i * in[channel_index].cols + j];
                }
            }
        }
        mean /= group_size * in[0].rows * in[0].cols;

        // Use a second pass to calculate the standard deviation for better stability
        matrix_float_t stdev = 0;
        for (int c = 0; c < group_size; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    matrix_float_t val = in[channel_index].data[i * in[channel_index].cols + j] - mean;
                    stdev += val * val;
                }
            }
        }
        stdev /= group_size * in[0].rows * in[0].cols;
        stdevs[g] = stdev; // Store the standard deviation for the backward pass

        // In a third pass, compute and store the output values
        for (int c = 0; c < group_size; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    matrix_float_t val = (in[channel_index].data[i * in[channel_index].cols + j] - mean) / stdev;
                    out[channel_index].data[i * out[channel_index].cols * j] = val;
                }
            }
        }
    }
}