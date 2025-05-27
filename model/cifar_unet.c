#include "../lib/bmp.h"
#include "../lib/cifar10.h"
#include "../lib/conv.h"
#include "../lib/norm.h"
#include "../lib/util.h"
#include "../lib/csv.h"
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

/**
 * Using the CIFAR-10 dataset:
 * Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. 
 * https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
 *
 * Model based on:
 * Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. 
 * https://doi.org/10.48550/arXiv.2006.11239
 */

const int IMAGE_HEIGHT = 32;
const int IMAGE_WIDTH = 32;
const int RESIZE_STRIDE = 2;
const int RESOLUTION_1_EMBED_DIM = 128;
const int RESOLUTION_2_EMBED_DIM = 256;
const int RESOLUTION_3_EMBED_DIM = 256;
const int RESOLUTION_4_EMBED_DIM = 256;
const int TIME_EMBED_DIM = 512;
const int KERNEL_SIZE = 3;
const int GROUP_SIZE = 32;
const int SELF_ATTENTION_KEY_DIM = 16;
const float DROPOUT_RATE = 0.1;

const int RESOLUTION_1_HEIGHT = IMAGE_HEIGHT;
const int RESOLUTION_1_WIDTH = IMAGE_WIDTH;
const int RESOLUTION_2_HEIGHT = (RESOLUTION_1_HEIGHT + RESIZE_STRIDE - 1) / RESIZE_STRIDE;
const int RESOLUTION_2_WIDTH = (RESOLUTION_1_WIDTH + RESIZE_STRIDE - 1) / RESIZE_STRIDE;
const int RESOLUTION_3_HEIGHT = (RESOLUTION_2_HEIGHT + RESIZE_STRIDE - 1) / RESIZE_STRIDE;
const int RESOLUTION_3_WIDTH = (RESOLUTION_2_WIDTH + RESIZE_STRIDE - 1) / RESIZE_STRIDE;
const int RESOLUTION_4_HEIGHT = (RESOLUTION_3_HEIGHT + RESIZE_STRIDE - 1) / RESIZE_STRIDE;
const int RESOLUTION_4_WIDTH = (RESOLUTION_3_WIDTH + RESIZE_STRIDE - 1) / RESIZE_STRIDE;

// Filepath string formats and lengths (including the \0 inserted by snprintf)
const char* DATA_PATH = "data/cifar_unet";
const int DATA_PATH_LENGTH = 16;
const char* DOWN_NAME_FORMAT = "/down_%d";
const int DOWN_NAME_FORMAT_LENGTH = 8;
const char* MID_NAME_FORMAT = "/mid";
const int MID_NAME_FORMAT_LENGTH = 5;
const char* UP_NAME_FORMAT = "/up_%d";
const int UP_NAME_FORMAT_LENGTH = 6;
const char* OUTPUT_NAME_FORMAT = "/output_conv.csv";
const int OUTPUT_NAME_FORMAT_LENGTH = 17;
const char* RESNET_NAME_FORMAT = "/resnet_%d";
const int RESNET_NAME_FORMAT_LENGTH = 10;
const char* CONV_NAME_FORMAT = "/conv_%d.csv";
const int CONV_NAME_FORMAT_LENGTH = 12;
const char* SELF_ATTENTION_NAME_FORMAT = "/self_attention_%d";
const int SELF_ATTENTION_NAME_FORMAT_LENGTH = 18;
const char* TIME_WEIGHT_FORMAT = "/time_weight.csv";
const int TIME_WEIGHT_FORMAT_LENGTH = 17;
const char* TIME_BIAS_FORMAT = "/time_bias.csv";
const int TIME_BIAS_FORMAT_LENGTH = 15;
const char* ATTENTION_QUERY_FORMAT = "/query.csv";
const int ATTENTION_QUERY_FORMAT_LENGTH = 11;
const char* ATTENTION_KEY_FORMAT = "/key.csv";
const int ATTENTION_KEY_FORMAT_LENGTH = 9;
const char* ATTENTION_VALUE_FORMAT = "/value.csv";
const int ATTENTION_VALUE_FORMAT_LENGTH = 11;
const char* ATTENTION_WEIGHT_FORMAT = "/weight.csv";
const int ATTENTION_WEIGHT_FORMAT_LENGTH = 12;
const char* ATTENTION_BIAS_FORMAT = "/bias.csv";
const int ATTENTION_BIAS_FORMAT_LENGTH = 10;

typedef struct ResnetBlockParams {
	Matrix** conv_1_kernels;
	Matrix** conv_2_kernels;
	Matrix* time_weights;
	Matrix* time_biases;
	Matrix** residual_conv_kernels;
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
	matrix_float_t* group_norm_means_1;
	matrix_float_t* group_norm_stdevs_1;
	Matrix* relu_1; // Group norm -> ReLU
	ConvData* conv_1;
	Matrix* time_dense; // With bias
	matrix_float_t* group_norm_means_2;
	matrix_float_t* group_norm_stdevs_2;
	Matrix* relu_2; // Group norm -> ReLU
	Matrix* dropout;
	ConvData* conv_2;
	ConvData* residual_conv;
	Matrix* result;
} ResnetBlockData;

typedef struct SelfAttentionData {
	Matrix* input;
	Matrix* Q_proj;
	Matrix* K_proj;
	Matrix* V_proj;
	Matrix* attention_weights;
	Matrix* attention;
	Matrix* product; // Without bias, unsummed
	Matrix* dense; // With bias
	Matrix* output; // Reshaped and residual added
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

	matrix_float_t* output_group_norm_means;
	matrix_float_t* output_group_norm_stdevs;
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

void multi_channel_relu(Matrix* X, int channels) {
	for (int i = 0; i < channels; i++) {
		relu(X[i].data, X[i].rows * X[i].cols);
	}
}

void _allocate_conv_kernels_and_data(Matrix** kernels, ConvData* d, int in_height, int in_width, int stride, int kernel_size, int in_channels, int out_channels) {
	for (int i = 0; i < out_channels; i++) {
		kernels[i] = malloc(in_channels * sizeof(Matrix));
		for (int j = 0; j < in_channels; j++) {
			kernels[i][j].rows = kernel_size;
			kernels[i][j].cols = kernel_size;
			kernels[i][j].data = malloc(kernel_size * kernel_size * sizeof(matrix_float_t));
		}
	}
	
	int out_height = (in_height + stride - 1) / stride;
	int out_width = (in_width + stride - 1) / stride;
	int num_convolutions = out_height * out_width;
	int conv_dim = kernel_size * kernel_size * in_channels;
	d->im2col = malloc(sizeof(Matrix));
	d->im2col->rows = num_convolutions;
	d->im2col->cols = conv_dim;
	d->im2col->data = malloc(num_convolutions * conv_dim * sizeof(matrix_float_t));

	d->kernel_matrix = malloc(sizeof(Matrix));
	d->kernel_matrix->rows = conv_dim;
	d->kernel_matrix->cols = out_channels;
	d->kernel_matrix->data = malloc(conv_dim * out_channels * sizeof(matrix_float_t));

	d->product = malloc(sizeof(Matrix));
	d->product->rows = num_convolutions;
	d->product->cols = out_channels;
	d->product->data = malloc(num_convolutions * out_channels * sizeof(matrix_float_t));

	d->output = malloc(out_channels * sizeof(Matrix));
	for (int i = 0; i < out_channels; i++) {
		d->output[i].rows = out_height;
		d->output[i].cols = out_width;
		d->output[i].data = malloc(out_height * out_width * sizeof(matrix_float_t));
	}
}

void _allocate_resnet_block(ResnetBlockParams* p, ResnetBlockData* d, int embed_dim, int height, int width, int in_channels, int kernel_size, int time_embed_dim, int group_size) {
	// Parameters
	// Conv kernels allocated later (with the conv data)
	p->conv_1_kernels = malloc(embed_dim * sizeof(Matrix*));
	p->conv_2_kernels = malloc(embed_dim * sizeof(Matrix*));
	p->residual_conv_kernels = malloc(embed_dim * sizeof(Matrix*));

	p->time_weights = malloc(sizeof(Matrix));
	p->time_weights->rows = time_embed_dim;
	p->time_weights->cols = embed_dim;
	p->time_weights->data = malloc(time_embed_dim * embed_dim * sizeof(matrix_float_t));

	p->time_biases = malloc(sizeof(Matrix));
	p->time_biases->rows = 1;
	p->time_biases->cols = embed_dim;
	p->time_biases->data = malloc(embed_dim * sizeof(matrix_float_t));

	// Data
	int num_groups_1 = (in_channels + group_size - 1) / group_size; // Round up just in case
	d->group_norm_means_1 = malloc(num_groups_1 * sizeof(matrix_float_t));
	d->group_norm_stdevs_1 = malloc(num_groups_1 * sizeof(matrix_float_t));
	d->relu_1 = malloc(in_channels * sizeof(Matrix));
	for (int i = 0; i < in_channels; i++) {
		d->relu_1[i].rows = height;
		d->relu_1[i].cols = width;
		d->relu_1[i].data = malloc(height * width * sizeof(matrix_float_t));
	}

	d->conv_1 = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->conv_1_kernels, d->conv_1, height, width, 1, kernel_size, in_channels, embed_dim);

	// d->time_product = malloc(sizeof(Matrix));
	// d->time_product->rows = time_embed_dim;
	// d->time_product->cols = embed_dim;
	// d->time_product->data = malloc(time_embed_dim * embed_dim * sizeof(matrix_float_t));

	d->time_dense = malloc(sizeof(Matrix));
	d->time_dense->rows = 1;
	d->time_dense->cols = embed_dim;
	d->time_dense->data = malloc(embed_dim * sizeof(matrix_float_t));

	int num_groups_2 = (embed_dim + group_size - 1) / group_size; // Round up just in case
	d->group_norm_means_2 = malloc(num_groups_2 * sizeof(matrix_float_t));
	d->group_norm_stdevs_2 = malloc(num_groups_2 * sizeof(matrix_float_t));

	// These three have the same dimensions, so just initialize them together
	d->relu_2 = malloc(embed_dim * sizeof(Matrix));
	d->dropout = malloc(embed_dim * sizeof(Matrix));
	d->result = malloc(embed_dim * sizeof(Matrix));
	for (int i = 0; i < embed_dim; i++) {
		d->relu_2[i].rows = height;
		d->relu_2[i].cols = width;
		d->relu_2[i].data = malloc(height * width * sizeof(matrix_float_t));

		d->dropout[i].rows = height;
		d->dropout[i].cols = width;
		d->dropout[i].data = malloc(height * width * sizeof(matrix_float_t));

		d->result[i].rows = height;
		d->result[i].cols = width;
		d->result[i].data = malloc(height * width * sizeof(matrix_float_t));
	}

	d->conv_2 = malloc(sizeof(ConvData));
	d->residual_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->conv_2_kernels, d->conv_2, height, width, 1, kernel_size, embed_dim, embed_dim);
	_allocate_conv_kernels_and_data(p->residual_conv_kernels, d->residual_conv, height, width, 1, 1, embed_dim, embed_dim);
}

void _allocate_self_attention_block(SelfAttentionParams* p, SelfAttentionData* d, int embed_dim, int key_dim, int height, int width) {
	// Parameters
	p->Q_proj = malloc(sizeof(Matrix));
	p->Q_proj->rows = embed_dim;
	p->Q_proj->cols = key_dim;
	p->Q_proj->data = malloc(embed_dim * key_dim * sizeof(matrix_float_t));

	p->K_proj = malloc(sizeof(Matrix));
	p->K_proj->rows = embed_dim;
	p->K_proj->cols = key_dim;
	p->K_proj->data = malloc(embed_dim * key_dim * sizeof(matrix_float_t));

	p->V_proj = malloc(sizeof(Matrix));
	p->V_proj->rows = embed_dim;
	p->V_proj->cols = key_dim;
	p->V_proj->data = malloc(embed_dim * key_dim * sizeof(matrix_float_t));

	p->weights = malloc(sizeof(Matrix));
	p->weights->rows = key_dim;
	p->weights->cols = embed_dim;
	p->weights->data = malloc(key_dim * embed_dim * sizeof(matrix_float_t));

	p->biases = malloc(sizeof(Matrix));
	p->biases->rows = 1;
	p->biases->cols = embed_dim;
	p->biases->data = malloc(embed_dim * sizeof(matrix_float_t));

	// Data
	int spatial_dim = height * width;
	d->input = malloc(sizeof(Matrix));
	d->input->rows = spatial_dim;
	d->input->cols = embed_dim;
	d->input->data = malloc(spatial_dim * embed_dim * sizeof(matrix_float_t));

	d->Q_proj = malloc(sizeof(Matrix));
	d->Q_proj->rows = spatial_dim;
	d->Q_proj->cols = key_dim;
	d->Q_proj->data = malloc(spatial_dim * key_dim * sizeof(matrix_float_t));

	d->K_proj = malloc(sizeof(Matrix));
	d->K_proj->rows = spatial_dim;
	d->K_proj->cols = key_dim;
	d->K_proj->data = malloc(spatial_dim * key_dim * sizeof(matrix_float_t));

	d->V_proj = malloc(sizeof(Matrix));
	d->V_proj->rows = spatial_dim;
	d->V_proj->cols = key_dim;
	d->V_proj->data = malloc(spatial_dim * key_dim * sizeof(matrix_float_t));

	d->attention_weights = malloc(sizeof(Matrix));
	d->attention_weights->rows = spatial_dim;
	d->attention_weights->cols = spatial_dim;
	d->attention_weights->data = malloc(spatial_dim * spatial_dim * sizeof(matrix_float_t));

	d->attention = malloc(sizeof(Matrix));
	d->attention->rows = spatial_dim;
	d->attention->cols = key_dim;
	d->attention->data = malloc(spatial_dim * key_dim * sizeof(matrix_float_t));

	d->product = malloc(sizeof(Matrix));
	d->product->rows = spatial_dim;
	d->product->cols = embed_dim;
	d->product->data = malloc(spatial_dim * embed_dim * sizeof(matrix_float_t));

	d->dense = malloc(sizeof(Matrix));
	d->dense->rows = spatial_dim;
	d->dense->cols = embed_dim;
	d->dense->data = malloc(spatial_dim * embed_dim * sizeof(matrix_float_t));

	d->output = malloc(embed_dim * sizeof(Matrix));
	for (int i = 0; i < embed_dim; i++) {
		d->output[i].rows = height;
		d->output[i].cols = width;
		d->output[i].data = malloc(spatial_dim * sizeof(matrix_float_t));
	}
}

void allocate_model_memory(ModelParams* p, ModelData* d) {
	// Input data
	d->X = malloc(3 * sizeof(Matrix));
	for (int i = 0; i < 3; i++) {
		d->X[i].rows = RESOLUTION_1_HEIGHT;
		d->X[i].cols = RESOLUTION_1_WIDTH;
		d->X[i].data = malloc(RESOLUTION_1_HEIGHT * RESOLUTION_1_WIDTH * sizeof(matrix_float_t));
	}
	d->time_embedding = malloc(sizeof(Matrix));
	d->time_embedding->rows = 1;
	d->time_embedding->cols = TIME_EMBED_DIM;
	d->time_embedding->data = malloc(TIME_EMBED_DIM * sizeof(matrix_float_t));

	// First downsampling layer: resnet block x2, downsampling convolution
	p->down_1_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->down_1_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_1_resnet_1, d->down_1_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 3, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_1_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->down_1_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_1_resnet_2, d->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_1_conv_kernels = malloc(RESOLUTION_2_EMBED_DIM * sizeof(Matrix*));
	d->down_1_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->down_1_conv_kernels, d->down_1_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// First downsampling layer: resnet block, self-attention block, resnet block, self-attention block, downsampling convolution
	p->down_2_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->down_2_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_2_resnet_1, d->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_2_self_attention_1 = malloc(sizeof(SelfAttentionParams));
	d->down_2_self_attention_1 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block(p->down_2_self_attention_1, d->down_2_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	p->down_2_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->down_2_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_2_resnet_2, d->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_2_self_attention_2 = malloc(sizeof(SelfAttentionParams));
	d->down_2_self_attention_2 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block(p->down_2_self_attention_2, d->down_2_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	p->down_2_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	d->down_2_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->down_2_conv_kernels, d->down_2_conv, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Third downsampling layer: resnet block x2 + downsampling convolution
	p->down_3_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->down_3_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_3_resnet_1, d->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_3_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->down_3_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_3_resnet_2, d->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_3_conv_kernels = malloc(RESOLUTION_4_EMBED_DIM * sizeof(Matrix*));
	d->down_3_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->down_3_conv_kernels, d->down_3_conv, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM);

	// Fourth downsampling layer: resnet block x2
	p->down_4_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->down_4_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_4_resnet_1, d->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->down_4_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->down_4_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->down_4_resnet_2, d->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);

	// Mid layer: resnet block, self-attention block, resnet block
	p->mid_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->mid_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->mid_resnet_1, d->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->mid_self_attention = malloc(sizeof(SelfAttentionParams));
	d->mid_self_attention = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block(p->mid_self_attention, d->mid_self_attention, RESOLUTION_4_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH);
	p->mid_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->mid_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->mid_resnet_2, d->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);

	// First upsampling layer: resnet block x2, nearest neighbours upscaling + a convolution
	p->up_1_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->up_1_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_1_resnet_1, d->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->up_1_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->up_1_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_1_resnet_2, d->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	d->up_1_nearest_neighbours = malloc(RESOLUTION_4_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_4_EMBED_DIM; i++) {
		d->up_1_nearest_neighbours[i].rows = RESOLUTION_3_HEIGHT;
		d->up_1_nearest_neighbours[i].cols = RESOLUTION_3_WIDTH;
		d->up_1_nearest_neighbours[i].data = malloc(RESOLUTION_3_HEIGHT * RESOLUTION_3_WIDTH * sizeof(matrix_float_t));
	}
	p->up_1_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	d->up_1_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->up_1_conv_kernels, d->up_1_conv, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, 1, KERNEL_SIZE, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Second upsampling layer: resnet block x2, nearest neighbours upscaling + a convolution
	p->up_2_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->up_2_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_2_resnet_1, d->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->up_2_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->up_2_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_2_resnet_2, d->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	d->up_2_nearest_neighbours = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_3_EMBED_DIM; i++) {
		d->up_2_nearest_neighbours[i].rows = RESOLUTION_2_HEIGHT;
		d->up_2_nearest_neighbours[i].cols = RESOLUTION_2_WIDTH;
		d->up_2_nearest_neighbours[i].data = malloc(RESOLUTION_2_HEIGHT * RESOLUTION_2_WIDTH * sizeof(matrix_float_t));
	}
	p->up_2_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	d->up_2_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->up_2_conv_kernels, d->up_2_conv, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, 1, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// Third upsampling layer: resnet block, self-attention block, resnet block, self-attention block, nearest neighbours upscaling + a convolution
	p->up_3_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->up_3_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_3_resnet_1, d->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->up_3_self_attention_1 = malloc(sizeof(SelfAttentionParams));
	d->up_3_self_attention_1 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block(p->up_3_self_attention_1, d->up_3_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	p->up_3_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->up_3_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_3_resnet_2, d->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->up_3_self_attention_2 = malloc(sizeof(SelfAttentionParams));
	d->up_3_self_attention_2 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block(p->up_3_self_attention_2, d->up_3_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	d->up_3_nearest_neighbours = malloc(RESOLUTION_2_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_2_EMBED_DIM; i++) {
		d->up_3_nearest_neighbours[i].rows = RESOLUTION_1_HEIGHT;
		d->up_3_nearest_neighbours[i].cols = RESOLUTION_1_WIDTH;
		d->up_3_nearest_neighbours[i].data = malloc(RESOLUTION_1_HEIGHT * RESOLUTION_1_WIDTH * sizeof(matrix_float_t));
	}
	p->up_3_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	d->up_3_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->up_3_conv_kernels, d->up_3_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 1, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);

	// Fourth upsampling layer: resnet block x2
	p->up_4_resnet_1 = malloc(sizeof(ResnetBlockParams));
	d->up_4_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_4_resnet_1, d->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);
	p->up_4_resnet_2 = malloc(sizeof(ResnetBlockParams));
	d->up_4_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block(p->up_4_resnet_2, d->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM, GROUP_SIZE);

	// Output layer: group normalization, non-linearity, and a final convolution
	int num_output_groups = (RESOLUTION_1_EMBED_DIM + GROUP_SIZE - 1) / GROUP_SIZE;
	d->output_group_norm_means = malloc(num_output_groups * sizeof(matrix_float_t));
	d->output_group_norm_stdevs = malloc(num_output_groups * sizeof(matrix_float_t));
	d->output_relu = malloc(RESOLUTION_1_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_1_EMBED_DIM; i++) {
		d->output_relu[i].rows = RESOLUTION_1_HEIGHT;
		d->output_relu[i].cols = RESOLUTION_1_WIDTH;
		d->output_relu[i].data = malloc(RESOLUTION_1_HEIGHT * RESOLUTION_1_WIDTH * sizeof(matrix_float_t));
	}
	p->output_conv_kernels = malloc(3 * sizeof(Matrix*));
	d->output_conv = malloc(sizeof(ConvData));
	_allocate_conv_kernels_and_data(p->output_conv_kernels, d->output_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 1, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, 3);
}

void _forward_attention(Matrix* X, SelfAttentionParams* params, SelfAttentionData* data) {
	// Computes unmasked scaled dot product attention
	int key_dimension = params->K_proj->cols;
	matrix_multiply_inplace(X, params->Q_proj, data->Q_proj);
	matrix_multiply_inplace(X, params->K_proj, data->K_proj);
	matrix_multiply_inplace(X, params->V_proj, data->V_proj);
	matrix_transpose(data->K_proj);
	matrix_multiply_inplace(data->Q_proj, data->K_proj, data->attention_weights);
	matrix_transpose(data->K_proj);
	matrix_scale(data->attention_weights, 1.0 / sqrt(key_dimension));
	softmax(data->attention_weights->data, data->attention_weights->rows, data->attention_weights->cols);
	matrix_multiply_inplace(data->attention_weights, data->V_proj, data->attention);
}

void _add_time_embedding(Matrix* X, Matrix* time_embedding, int channels) {
	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < X[c].rows * X[c].cols; i++) {
			X[c].data[i] += time_embedding->data[c];
		}
	}
}

void _dropout(Matrix* X, Matrix* Y, int channels) {
	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < X[c].rows * X[c].cols; i++) {
			if ((float) rand() / RAND_MAX < DROPOUT_RATE) {
				Y[c].data[i] = 0;
			} else {
				Y[c].data[i] = X[c].data[i];
			}
		}
	}
}

void _forward_resnet(Matrix* X, Matrix* time_emb, ResnetBlockParams* p, ResnetBlockData* d, int in_channels, int out_channels, int group_size) {
	// First chunk
	group_norm(X, d->relu_1, d->group_norm_stdevs_1, d->group_norm_means_1, in_channels, group_size);
	multi_channel_relu(d->relu_1, in_channels);
	conv(d->relu_1, p->conv_1_kernels, d->conv_1, in_channels, out_channels, 1);

	// Time embedding
	matrix_multiply_inplace(time_emb, p->time_weights, d->time_dense);
	matrix_add(d->time_dense, p->time_biases);
	_add_time_embedding(d->conv_1->output, time_emb, out_channels);

	// Second chunk
	group_norm(d->conv_1->output, d->relu_2, d->group_norm_stdevs_2, d->group_norm_means_2, out_channels, group_size);
	multi_channel_relu(d->relu_2, out_channels);
	_dropout(d->relu_2, d->dropout, out_channels);
	conv(d->dropout, p->conv_2_kernels, d->conv_2, out_channels, out_channels, 1);

	// Skip connection
	Matrix* residual = X;
	if (in_channels != out_channels) {
		conv(X, p->residual_conv_kernels, d->residual_conv, in_channels, out_channels, 1);
		residual = d->residual_conv->output;
	}
	for (int c = 0; c < out_channels; c++) {
		for (int i = 0; i < d->result->rows * d->result->cols; i++) {
			d->result[c].data[i] = d->conv_2->output[c].data[i] + residual[c].data[i];
		}
	}
}

void _nearest_neighbours(Matrix* in, Matrix* out, int channels, int scale) {
	int in_width = in[0].cols;
	int out_height = out[0].rows;
	int out_width = out[0].cols;

	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < out_height; i++) {
			for (int j = 0; j < out_width; j++) {
				out[c].data[i * out_width + j] = in[c].data[(i / scale) * in_width + j / scale];
			}
		}
	}
}

void forward(ModelParams* p, ModelData* d) {
	Matrix* next;

	// Down
	_forward_resnet(d->X, d->time_embedding, p->down_1_resnet_1, d->down_1_resnet_1, 3, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->down_1_resnet_1->result, d->time_embedding, p->down_1_resnet_2, d->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	conv(d->down_1_resnet_2->result, p->down_1_conv_kernels, d->down_1_conv, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM, RESIZE_STRIDE);

	_forward_resnet(d->down_1_conv->output, d->time_embedding, p->down_2_resnet_1, d->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->down_2_resnet_1->result, p->down_2_self_attention_1, d->down_2_self_attention_1);
	_forward_resnet(d->down_2_self_attention_1->output, d->time_embedding, p->down_2_resnet_2, d->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->down_2_resnet_2->result, p->down_2_self_attention_2, d->down_2_self_attention_2);
	conv(d->down_2_self_attention_2->output, p->down_2_conv_kernels, d->down_2_conv, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM, RESIZE_STRIDE);

	_forward_resnet(d->down_2_conv->output, d->time_embedding, p->down_3_resnet_1, d->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->down_3_resnet_1->result, d->time_embedding, p->down_3_resnet_2, d->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	conv(d->down_3_resnet_2->result, p->down_3_conv_kernels, d->down_3_conv, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM, RESIZE_STRIDE);

	_forward_resnet(d->down_3_conv->output, d->time_embedding, p->down_4_resnet_1, d->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->down_4_resnet_1->result, d->time_embedding, p->down_4_resnet_2, d->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);

	// Mid
	_forward_resnet(d->down_4_resnet_2->result, d->time_embedding, p->mid_resnet_1, d->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->mid_resnet_1->result, p->mid_self_attention, d->mid_self_attention);
	_forward_resnet(d->mid_self_attention->output, d->time_embedding, p->mid_resnet_2, d->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);

	// Up
	_forward_resnet(d->mid_resnet_2->result, d->time_embedding, p->up_1_resnet_1, d->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_1_resnet_1->result, d->time_embedding, p->up_1_resnet_2, d->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_nearest_neighbours(d->up_1_resnet_2->result, d->up_1_nearest_neighbours, RESOLUTION_4_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_1_nearest_neighbours;
	if (RESOLUTION_4_EMBED_DIM != RESOLUTION_3_EMBED_DIM) {
		conv(d->up_1_nearest_neighbours, p->up_1_conv_kernels, d->up_1_conv, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM, 1);
		next = d->up_1_conv->output;
	}
	
	_forward_resnet(next, d->time_embedding, p->up_2_resnet_1, d->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_2_resnet_1->result, d->time_embedding, p->up_2_resnet_2, d->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_nearest_neighbours(d->up_2_resnet_2->result, d->up_2_nearest_neighbours, RESOLUTION_3_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_2_nearest_neighbours;
	if (RESOLUTION_3_EMBED_DIM != RESOLUTION_2_EMBED_DIM) {
		conv(d->up_2_nearest_neighbours, p->up_2_conv_kernels, d->up_2_conv, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM, 1);
		next = d->up_2_conv->output;
	}
	
	_forward_resnet(next, d->time_embedding, p->up_3_resnet_1, d->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->up_3_resnet_1->result, p->up_3_self_attention_1, d->up_3_self_attention_1);
	_forward_resnet(d->up_3_self_attention_1->output, d->time_embedding, p->up_3_resnet_2, d->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->up_3_resnet_2->result, p->up_3_self_attention_1, d->up_3_self_attention_1);
	_nearest_neighbours(d->up_3_self_attention_2->output, d->up_3_nearest_neighbours, RESOLUTION_2_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_3_nearest_neighbours;
	if (RESOLUTION_2_EMBED_DIM != RESOLUTION_1_EMBED_DIM) {
		conv(d->up_3_nearest_neighbours, p->up_3_conv_kernels, d->up_3_conv, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM, 1);
		next = d->up_3_conv->output;
	}
	
	_forward_resnet(next, d->time_embedding, p->up_4_resnet_1, d->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_4_resnet_1->result, d->time_embedding, p->up_4_resnet_2, d->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);

	// Output
	group_norm(d->up_4_resnet_2->result, d->output_relu, d->output_group_norm_stdevs, d->output_group_norm_means, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	multi_channel_relu(d->output_relu, RESOLUTION_1_EMBED_DIM);
	conv(d->output_relu, p->output_conv_kernels, d->output_conv, RESOLUTION_1_EMBED_DIM, 3, 1);
}

// For parameters with ReLU non-linearity
void _init_params_he(Matrix* m, int fan_in) {
	double scale = sqrt(6.0 / fan_in);
	for (int i = 0; i < m->rows * m->cols; i++) {
		m->data[i] = 2 * scale * (double) rand() / RAND_MAX - scale;
	}
}

// For parameters with softmax non-linearity
void _init_params_xavier(Matrix* m, int fan_in, int fan_out) {
	double scale = sqrt(6.0 / (fan_in + fan_out));
	for (int i = 0; i < m->rows * m->cols; i++) {
		m->data[i] = 2 * scale * (double) rand() / RAND_MAX - scale;
	}
}

void _init_conv_kernels(Matrix** kernels, int height, int width, int in_channels, int out_channels) {
	int fan_in = height * width;
	for (int i = 0; i < out_channels; i++) {
		for (int j = 0; j < in_channels; j++) {
			_init_params_he(&kernels[i][j], fan_in);
		}
	}
}

void _init_resnet_block(ResnetBlockParams* p, int height, int width, int in_channels, int out_channels, int time_embed_dim) {
	_init_conv_kernels(p->conv_1_kernels, height, width, in_channels, out_channels);
	_init_conv_kernels(p->conv_2_kernels, height, width, out_channels, out_channels);
	_init_params_he(p->time_weights, time_embed_dim);
	for (int i = 0; i < p->time_biases->cols; i++) {
		p->time_biases->data[i] = 0;
	}
	_init_conv_kernels(p->residual_conv_kernels, height, width, in_channels, out_channels);
}

void _init_self_attention_block(SelfAttentionParams* p, int height, int width, int key_dim) {
	int fan_in = height * width;
	_init_params_xavier(p->Q_proj, fan_in, key_dim);
	_init_params_xavier(p->K_proj, fan_in, key_dim);
	_init_params_he(p->V_proj, fan_in);
	_init_params_he(p->weights, key_dim);
	for (int i = 0; i < p->biases->cols; i++) {
		p->biases->data[i] = 0;
	}
}

void _save_matrix(Matrix* m, char* filepath) {
	float* buffer = malloc(m->rows * m->cols * sizeof(float));
	for (int i = 0; i < m->rows * m->cols; i++) {
		buffer[i] = (float) m->data[i];
	}
	write_csv_contents(filepath, buffer, m->cols, m->rows);
	free(buffer);
}

void _save_conv_kernels(Matrix** kernels, int in_channels, int out_channels, char* filepath) {
	int kernel_size = kernels[0][0].rows;
	int kernel_area = kernel_size * kernel_size;
	float* data_buffer = malloc(in_channels * out_channels * kernel_area * sizeof(float));

	for (int i = 0; i < out_channels; i++) {
		for (int j = 0; j < in_channels; j++) {
			for (int k = 0; k < kernel_area; k++) {
				data_buffer[i * in_channels * kernel_area + j * kernel_area + k] = (float) kernels[i][j].data[k];
			}
		}
	}

	write_csv_contents(filepath, data_buffer, kernel_area, in_channels * out_channels);

	free(data_buffer);
}

void _save_resnet_block(ResnetBlockParams* p, int in_channels, int out_channels, char* buffer, int buffer_position) {
	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 1);
	_save_conv_kernels(p->conv_1_kernels, in_channels, out_channels, buffer);
	
	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 2);
	_save_conv_kernels(p->conv_2_kernels, out_channels, out_channels, buffer);
	
	snprintf(&buffer[buffer_position], TIME_WEIGHT_FORMAT_LENGTH, "%s", TIME_WEIGHT_FORMAT);
	_save_matrix(p->time_weights, buffer);
	
	snprintf(&buffer[buffer_position], TIME_BIAS_FORMAT_LENGTH, "%s", TIME_BIAS_FORMAT);
	_save_matrix(p->time_biases, buffer);

	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 3);
	_save_conv_kernels(p->residual_conv_kernels, in_channels, out_channels, buffer);
}

void _save_self_attention_block(SelfAttentionParams* p, char* buffer, int buffer_position) {
	snprintf(&buffer[buffer_position], ATTENTION_QUERY_FORMAT_LENGTH, "%s", ATTENTION_QUERY_FORMAT);
	_save_matrix(p->Q_proj, buffer);
	
	snprintf(&buffer[buffer_position], ATTENTION_KEY_FORMAT_LENGTH, "%s", ATTENTION_KEY_FORMAT);
	_save_matrix(p->K_proj, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_VALUE_FORMAT_LENGTH, "%s", ATTENTION_VALUE_FORMAT);
	_save_matrix(p->V_proj, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_WEIGHT_FORMAT_LENGTH, "%s", ATTENTION_WEIGHT_FORMAT);
	_save_matrix(p->weights, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_BIAS_FORMAT_LENGTH, "%s", ATTENTION_BIAS_FORMAT);
	_save_matrix(p->biases, buffer);
}

void save_parameters(ModelParams* p) {
	char filepath_name_buffer[255];
	snprintf(filepath_name_buffer, DATA_PATH_LENGTH, "%s", DATA_PATH);
	mkdir(filepath_name_buffer, 0777);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_1_resnet_1, 3, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_1_resnet_2, 3, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->down_1_conv_kernels, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_self_attention_block(p->down_2_self_attention_1, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_self_attention_block(p->down_2_self_attention_2, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->down_2_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 3);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->down_3_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 4);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], MID_NAME_FORMAT_LENGTH, "%s", MID_NAME_FORMAT);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 0);
	mkdir(filepath_name_buffer, 0777);
	_save_self_attention_block(p->mid_self_attention, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->up_1_conv_kernels, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->up_2_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 3);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_self_attention_block(p->up_3_self_attention_1, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_self_attention_block(p->up_3_self_attention_2, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_save_conv_kernels(p->up_3_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 4);
	mkdir(filepath_name_buffer, 0777);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	mkdir(filepath_name_buffer, 0777);
	_save_resnet_block(p->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], OUTPUT_NAME_FORMAT_LENGTH, "%s", OUTPUT_NAME_FORMAT);
	_save_conv_kernels(p->output_conv_kernels, RESOLUTION_1_EMBED_DIM, 3, filepath_name_buffer);
}

void _load_matrix(Matrix* m, char* filepath) {
	float* buffer = read_csv_contents(filepath);
	for (int i = 0; i < m->rows * m->cols; i++) {
		m->data[i] = (matrix_float_t) buffer[i];
	}
	free(buffer);
}

void _load_conv_kernels(Matrix** kernels, int in_channels, int out_channels, char* filepath) {
	int kernel_size = kernels[0][0].rows;
	int kernel_area = kernel_size * kernel_size;
	float* data_buffer = read_csv_contents(filepath);

	for (int i = 0; i < out_channels; i++) {
		for (int j = 0; j < in_channels; j++) {
			for (int k = 0; k < kernel_area; k++) {
				kernels[i][j].data[k] = (matrix_float_t) data_buffer[i * in_channels * kernel_area + j * kernel_area + k];
			}
		}
	}

	free(data_buffer);
}

void _load_resnet_block(ResnetBlockParams* p, int in_channels, int out_channels, char* buffer, int buffer_position) {
	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 1);
	_load_conv_kernels(p->conv_1_kernels, in_channels, out_channels, buffer);
	
	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 2);
	_load_conv_kernels(p->conv_2_kernels, out_channels, out_channels, buffer);
	
	snprintf(&buffer[buffer_position], TIME_WEIGHT_FORMAT_LENGTH, "%s", TIME_WEIGHT_FORMAT);
	_load_matrix(p->time_weights, buffer);
	
	snprintf(&buffer[buffer_position], TIME_BIAS_FORMAT_LENGTH, "%s", TIME_BIAS_FORMAT);
	_load_matrix(p->time_biases, buffer);

	snprintf(&buffer[buffer_position], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 3);
	_load_conv_kernels(p->residual_conv_kernels, in_channels, out_channels, buffer);
}

void _load_self_attention_block(SelfAttentionParams* p, char* buffer, int buffer_position) {
	snprintf(&buffer[buffer_position], ATTENTION_QUERY_FORMAT_LENGTH, "%s", ATTENTION_QUERY_FORMAT);
	_load_matrix(p->Q_proj, buffer);
	
	snprintf(&buffer[buffer_position], ATTENTION_KEY_FORMAT_LENGTH, "%s", ATTENTION_KEY_FORMAT);
	_load_matrix(p->K_proj, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_VALUE_FORMAT_LENGTH, "%s", ATTENTION_VALUE_FORMAT);
	_load_matrix(p->V_proj, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_WEIGHT_FORMAT_LENGTH, "%s", ATTENTION_WEIGHT_FORMAT);
	_load_matrix(p->weights, buffer);

	snprintf(&buffer[buffer_position], ATTENTION_BIAS_FORMAT_LENGTH, "%s", ATTENTION_BIAS_FORMAT);
	_load_matrix(p->biases, buffer);
}

void load_parameters(ModelParams* p) {
	char filepath_name_buffer[255];
	snprintf(filepath_name_buffer, DATA_PATH_LENGTH, "%s", DATA_PATH);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 1);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->down_1_resnet_1, 3, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->down_1_resnet_2, 3, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->down_1_conv_kernels, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 2);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 1);
	_load_self_attention_block(p->down_2_self_attention_1, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 2);
	_load_self_attention_block(p->down_2_self_attention_2, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->down_2_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->down_3_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], DOWN_NAME_FORMAT_LENGTH, DOWN_NAME_FORMAT, 4);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + DOWN_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], MID_NAME_FORMAT_LENGTH, "%s", MID_NAME_FORMAT);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 0);
	_load_self_attention_block(p->mid_self_attention, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + MID_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 1);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->up_1_conv_kernels, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 2);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->up_2_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer);
	
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 1);
	_load_self_attention_block(p->up_3_self_attention_1, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], SELF_ATTENTION_NAME_FORMAT_LENGTH, SELF_ATTENTION_NAME_FORMAT, 2);
	_load_self_attention_block(p->up_3_self_attention_2, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + SELF_ATTENTION_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], CONV_NAME_FORMAT_LENGTH, CONV_NAME_FORMAT, 0);
	_load_conv_kernels(p->up_3_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], UP_NAME_FORMAT_LENGTH, UP_NAME_FORMAT, 4);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 1);
	_load_resnet_block(p->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);
	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH - 2], RESNET_NAME_FORMAT_LENGTH, RESNET_NAME_FORMAT, 2);
	_load_resnet_block(p->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, filepath_name_buffer, DATA_PATH_LENGTH + UP_NAME_FORMAT_LENGTH + RESNET_NAME_FORMAT_LENGTH - 3);

	snprintf(&filepath_name_buffer[DATA_PATH_LENGTH - 1], OUTPUT_NAME_FORMAT_LENGTH, "%s", OUTPUT_NAME_FORMAT);
	_load_conv_kernels(p->output_conv_kernels, RESOLUTION_1_EMBED_DIM, 3, filepath_name_buffer);
}

void init_parameters(ModelParams* p) {
	_init_resnet_block(p->down_1_resnet_1, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 3, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->down_1_resnet_2, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->down_1_conv_kernels, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	
	_init_resnet_block(p->down_2_resnet_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->down_2_self_attention_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_resnet_block(p->down_2_resnet_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->down_2_self_attention_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_conv_kernels(p->down_2_conv_kernels, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	
	_init_resnet_block(p->down_3_resnet_1, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->down_3_resnet_2, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->down_3_conv_kernels, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	
	_init_resnet_block(p->down_4_resnet_1, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->down_4_resnet_2, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);

	_init_resnet_block(p->mid_resnet_1, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->mid_self_attention, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_resnet_block(p->mid_resnet_2, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	
	_init_resnet_block(p->up_1_resnet_1, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_1_resnet_2, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->up_1_conv_kernels, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	
	_init_resnet_block(p->up_2_resnet_1, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_2_resnet_2, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->up_2_conv_kernels, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	
	_init_resnet_block(p->up_3_resnet_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->up_3_self_attention_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_resnet_block(p->up_3_resnet_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->up_3_self_attention_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_conv_kernels(p->up_3_conv_kernels, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);

	_init_resnet_block(p->up_4_resnet_1, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_4_resnet_2, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);

	_init_conv_kernels(p->output_conv_kernels, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, 3);
}

void init() {
	ModelParams p;
	ModelData d;

	allocate_model_memory(&p, &d);

	init_parameters(&p);

	save_parameters(&p);
}

void train(int num_epochs) {
	(void) num_epochs;

	int data_fds[5];
	data_fds[0] = open("data/cifar/data_batch_1.bin", O_RDONLY);
	data_fds[1] = open("data/cifar/data_batch_2.bin", O_RDONLY);
	data_fds[2] = open("data/cifar/data_batch_3.bin", O_RDONLY);
	data_fds[3] = open("data/cifar/data_batch_4.bin", O_RDONLY);
	data_fds[4] = open("data/cifar/data_batch_5.bin", O_RDONLY);

	ModelParams p;
	ModelData d;
	allocate_model_memory(&p, &d);

	// Speed up testing by avoiding file system operations
	// load_parameters(&p);
	init_parameters(&p);

	load_example(d.X, data_fds[0]);

	forward(&p, &d);

	for (int i = 0; i < 3; i++) {
		print_matrix(*d.output_conv->output);
	}

	for (int i = 0; i < 5; i++) {
		close(data_fds[i]);
	}
}

void run(int num_predictions) {
	(void) num_predictions;
}

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