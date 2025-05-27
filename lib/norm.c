#include "norm.h"

void group_norm(Matrix* in, Matrix* out, matrix_float_t* stdevs, matrix_float_t* means, int channels, int group_size) {
    int epsilon = 1e-8;
    int num_groups = (channels + group_size - 1) / group_size;
    for (int g = 0; g < num_groups; g++) {
        int num_in_this_group = channels - g * group_size;
        if (num_in_this_group > group_size) {
            num_in_this_group = group_size;
        }

        matrix_float_t mean = 0;
        for (int c = 0; c < num_in_this_group; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    mean += in[channel_index].data[i * in[channel_index].cols + j];
                }
            }
        }
        mean /= num_in_this_group * in[0].rows * in[0].cols;
        means[g] = mean; // Store the mean for the backward pass

        // Use a second pass to calculate the standard deviation for better stability
        matrix_float_t stdev = 0;
        for (int c = 0; c < num_in_this_group; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    matrix_float_t val = in[channel_index].data[i * in[channel_index].cols + j] - mean;
                    stdev += val * val;
                }
            }
        }
        stdev /= num_in_this_group * in[0].rows * in[0].cols;
        stdevs[g] = stdev; // Store the standard deviation for the backward pass

        // In a third pass, compute and store the output values
        for (int c = 0; c < num_in_this_group; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < in[channel_index].rows; i++) {
                for (int j = 0; j < in[channel_index].cols; j++) {
                    matrix_float_t val = (in[channel_index].data[i * in[channel_index].cols + j] - mean) / (stdev + epsilon);
                    out[channel_index].data[i * out[channel_index].cols + j] = val;
                }
            }
        }
    }
}