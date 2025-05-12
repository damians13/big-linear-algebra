#include "../lib/bmp.h"
#include "../lib/cifar10.h"
#include "../lib/conv.h"
#include "../lib/util.h"
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <string.h>

/**
 * Using the CIFAR-10 dataset:
 * Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. 
 * https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
 *
 * Model based on:
 * Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. 
 * https://doi.org/10.48550/arXiv.2006.11239
 */

const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;

typedef struct ResnetBlockParams {
	Matrix** conv_1_kernels;
	Matrix** conv_2_kernels;
	Matrix* time_weights;
	Matrix* time_biases;
	Matrix* residual_weights;
	Matrix* residual_biases;
} ResnetBlockParams;

typedef struct SelfAttentionParams {
	Matrix* Q_proj;
	Matrix* K_proj;
	Matrix* V_proj;
	Matrix* weights;
	Matrix* biases;
} SelfAttentionParams;

typedef struct ModelParams {
	ResnetBlockParams* down_1_resnet_1;
	ResnetBlockParams* down_1_resnet_2;
	Matrix** down_1_conv_kernels; // Downsampling layer 1, downsampling convolution

	ResnetBlockParams* down_2_resnet_1;
	SelfAttentionParams* down_2_self_attention_1;
	ResnetBlockParams* down_2_resnet_2;
	SelfAttentionParams* down_2_self_attention_2;
	Matrix** down_2_conv_kernels; // Downsampling layer 2, downsampling convolution parameters

	ResnetBlockParams* down_3_resnet_1;
	ResnetBlockParams* down_3_resnet_2;
	Matrix** down_3_conv_kernels; // Downsampling layer 3, downsampling convolution
	
	ResnetBlockParams* down_4_resnet_1;
	ResnetBlockParams* down_4_resnet_2;
	Matrix** down_4_conv_kernels; // Downsampling layer 4, downsampling convolution

	ResnetBlockParams* mid_resnet_1;
	SelfAttentionParams* mid_self_attention;
	ResnetBlockParams* mid_resnet_2;

	ResnetBlockParams* up_1_resnet_1;
	ResnetBlockParams* up_1_resnet_2;
	Matrix** up_1_conv_kernels; // Upsampling layer 1 convolution

	ResnetBlockParams* up_2_resnet_1;
	ResnetBlockParams* up_2_resnet_2;
	Matrix** up_2_conv_kernels; // Upsampling layer 2 convolution

	ResnetBlockParams* up_3_resnet_1;
	SelfAttentionParams* up_3_self_attention_1;
	ResnetBlockParams* up_3_resnet_2;
	SelfAttentionParams* up_3_self_attention_2;
	Matrix** up_3_conv_kernels; // Upsampling layer 3 convolution

	ResnetBlockParams* up_4_resnet_1;
	ResnetBlockParams* up_4_resnet_2;

	Matrix** output_conv_kernels; // Final convolution
} ModelParams;

typedef struct ResnetBlockData {
	Matrix* group_norm_1;
	Matrix* relu_1;
	ConvData* conv_1;
	Matrix* time_product; // Without bias, unsummed
	Matrix* time_dense; // With bias
	Matrix* concat_time;
	Matrix* group_norm_2;
	Matrix* relu_2;
	Matrix* dropout;
	ConvData* conv_2;
	Matrix* residual_product;
	Matrix* residual_dense;
	Matrix* result;
} ResnetBlockData;

typedef struct SelfAttentionData {
	Matrix* Q_proj;
	Matrix* K_proj;
	Matrix* V_proj;
	Matrix* attention;
	Matrix* result;
} SelfAttentionData;

typedef struct ModelData {
	Matrix* X;
	Matrix* time_embedding; // Passed through ReLU already

	ResnetBlockData* down_1_resnet_1;
	ResnetBlockData* down_1_resnet_2;
	ConvData* down_1_conv;

	ResnetBlockData* down_2_resnet_1;
	SelfAttentionData* down_2_self_attention_1;
	ResnetBlockData* down_2_resnet_2;
	SelfAttentionData* down_2_self_attention_2;
	ConvData* down_2_conv;
	
	ResnetBlockData* down_3_resnet_1;
	ResnetBlockData* down_3_resnet_2;
	ConvData* down_3_conv;
	
	ResnetBlockData* down_4_resnet_1;
	ResnetBlockData* down_4_resnet_2;
	ConvData* down_4_conv;

	ResnetBlockData* mid_resnet_1;
	SelfAttentionData* mid_self_attention;
	ResnetBlockData* mid_resnet_2;
	
	ResnetBlockData* up_1_resnet_1;
	ResnetBlockData* up_1_resnet_2;
	Matrix* up_1_nearest_neighbours;
	ConvData* up_1_conv;
	
	ResnetBlockData* up_2_resnet_1;
	ResnetBlockData* up_2_resnet_2;
	Matrix* up_2_nearest_neighbours;
	ConvData* up_2_conv;
	
	ResnetBlockData* up_3_resnet_1;
	SelfAttentionData* up_3_self_attention_1;
	ResnetBlockData* up_3_resnet_2;
	SelfAttentionData* up_3_self_attention_2;
	Matrix* up_3_nearest_neighbours;
	ConvData* up_3_conv;
	
	ResnetBlockData* up_4_resnet_1;
	ResnetBlockData* up_4_resnet_2;

	Matrix* output_group_norm;
	Matrix* output_relu;
	ConvData* output_conv;
} ModelData;

void load_example(Matrix* x, int fd) {
	uint8_t data[3072];
	fill_random_data(fd, data);

	// Map pixel values from [0, 255] to [-1, 1] and convert to floats
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < IMAGE_HEIGHT; i++) {
			for (int j = 0; j < IMAGE_WIDTH; j++) {
				x[c].data[i * IMAGE_WIDTH + j] = ((matrix_float_t) data[c * IMAGE_HEIGHT * IMAGE_WIDTH + i * IMAGE_WIDTH + j] - 127.5) / 127.5;
			}
		}
	}
}

void compute_attention(Matrix* X, SelfAttentionParams* params, SelfAttentionData* data) {
	// Computes unmasked scaled dot product attention
	int key_dimension = params->K_proj->cols;
	matrix_multiply_inplace(X, params->Q_proj, data->Q_proj);
	matrix_multiply_inplace(X, params->K_proj, data->K_proj);
	matrix_multiply_inplace(X, params->V_proj, data->V_proj);
	matrix_transpose(data->K_proj);
	matrix_multiply_inplace(data->Q_proj, data->K_proj, data->attention);
	matrix_transpose(data->K_proj);
	matrix_scale(data->attention, 1.0 / sqrt(key_dimension));
	softmax(data->attention->data, data->attention->rows, data->attention->cols);
	matrix_multiply_inplace(data->attention, data->V_proj, data->result);
}

void init() {}

void train(int num_epochs) {
	int data_fds[5];
	data_fds[0] = open("data/cifar/data_batch_1.bin", O_RDONLY);
	data_fds[1] = open("data/cifar/data_batch_2.bin", O_RDONLY);
	data_fds[2] = open("data/cifar/data_batch_3.bin", O_RDONLY);
	data_fds[3] = open("data/cifar/data_batch_4.bin", O_RDONLY);
	data_fds[4] = open("data/cifar/data_batch_5.bin", O_RDONLY);

	Matrix x[3];
	x[0] = (Matrix) { IMAGE_HEIGHT, IMAGE_WIDTH, malloc(IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(matrix_float_t)) };
	x[1] = (Matrix) { IMAGE_HEIGHT, IMAGE_WIDTH, malloc(IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(matrix_float_t)) };
	x[2] = (Matrix) { IMAGE_HEIGHT, IMAGE_WIDTH, malloc(IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(matrix_float_t)) };

	load_example(x, data_fds[0]);

	for (int i = 0; i < 5; i++) {
		close(data_fds[i]);
	}
}

void run(int num_predictions) {}

int main(int argc, char** argv) {
	srand(42);
	if (argc < 2) {
		printf("Please supply an argument, options:\n\trun [<num samples> (default 1)]\n\ttrain <num epochs>\n\tinit\n");
		exit(1);
	}
	if (strncmp(argv[1], "run", 3) == 0) {
		if (argc < 3) {	
			run(1);
		} else {
			run(atoi(argv[2]));
		}
	} else if (strncmp(argv[1], "train", 5) == 0) {
		if (argc < 3) {
			printf("Please supply a number of epochs, usage:\n\ttrain <num_epochs>\n");
			exit(1);
		}
		train(atoi(argv[2]));
	} else if (strncmp(argv[1], "init", 4) == 0) {
		init();
	} else {
		printf("Unrecognized argument, options:\n\trun [<num samples> (default 1)]\n\ttrain <num epochs>\n\tinit\n");
		exit(1);
	}
}