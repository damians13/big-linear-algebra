#include "norm.h"

const int epsilon = 1e-8;

void group_norm(Matrix* in, Matrix* out, matrix_float_t* stdevs, matrix_float_t* means, int channels, int group_size) {
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

void group_norm_ddx(Matrix* source, Matrix* dest, Matrix* data, matrix_float_t* means, matrix_float_t* stdevs, int channels, int group_size) {
    int num_groups = (channels + group_size - 1) / group_size;
    int height = source[0].rows;
    int width = source[0].cols;
    for (int g = 0; g < num_groups; g++) {
        int num_in_this_group = channels - g * group_size;
        if (num_in_this_group > group_size) {
            num_in_this_group = group_size;
        }

        matrix_float_t group_gradient_sum = 0;
        matrix_float_t group_gradient_weighted_sum = 0;

        for (int c = 0; c < num_in_this_group; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    matrix_float_t weight = (data[channel_index].data[i * width + j] - means[g]) / (stdevs[g] + epsilon);
                    group_gradient_sum += source[channel_index].data[i * width + j];
                    group_gradient_weighted_sum += weight * source[channel_index].data[i * width + j];
                }
            }
        }

        group_gradient_sum /= num_in_this_group * height * width;
        group_gradient_weighted_sum /= num_in_this_group * height * width;

        for (int c = 0; c < num_in_this_group; c++) {
            int channel_index = g * group_size + c;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    matrix_float_t normalized_value = (data[channel_index].data[i * width + j] - means[g]) / (stdevs[g] + epsilon);
                    dest[channel_index].data[i * width + j] = (
                        source[channel_index].data[i * width + j]
                        - group_gradient_sum
                        - normalized_value * group_gradient_weighted_sum
                        ) / (stdevs[g] + epsilon);
                }
            }
        }
    }
}