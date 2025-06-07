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
	Matrix* attention_weights_raw;
	Matrix* attention_weights;
	Matrix* attention;
	Matrix* product; // NOTE: Only used in backward pass, don't touch in forward pass
	Matrix* dense;
	Matrix* output; // Reshaped
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
	
	Matrix* up_1_input_concat_skip;
	ResnetBlockData* up_1_resnet_1;
	ResnetBlockData* up_1_resnet_2;
	Matrix* up_1_nearest_neighbours;
	ConvData* up_1_conv;
	
	Matrix* up_2_input_concat_skip;
	ResnetBlockData* up_2_resnet_1;
	ResnetBlockData* up_2_resnet_2;
	Matrix* up_2_nearest_neighbours;
	ConvData* up_2_conv;
	
	Matrix* up_3_input_concat_skip;
	ResnetBlockData* up_3_resnet_1;
	SelfAttentionData* up_3_self_attention_1;
	ResnetBlockData* up_3_resnet_2;
	SelfAttentionData* up_3_self_attention_2;
	Matrix* up_3_nearest_neighbours;
	ConvData* up_3_conv;
	
	Matrix* up_4_input_concat_skip;
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

void multi_channel_relu_ddx(Matrix* source, Matrix* dest, Matrix* relu_result, int channels) {
	int height = dest[0].rows;
	int width = dest[0].cols;
	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < height * width; i++) {
			if (relu_result[c].data[i] <= 0) {
				dest[c].data[i] = 0;
			} else {
				dest[c].data[i] = source[c].data[i];
			}
		}
	}
}

void _allocate_conv_kernels(Matrix** kernels, int kernel_size, int in_channels, int out_channels) {
	for (int i = 0; i < out_channels; i++) {
		kernels[i] = malloc(in_channels * sizeof(Matrix));
		for (int j = 0; j < in_channels; j++) {
			kernels[i][j].rows = kernel_size;
			kernels[i][j].cols = kernel_size;
			kernels[i][j].data = malloc(kernel_size * kernel_size * sizeof(matrix_float_t));
		}
	}
}

void _allocate_conv_data(ConvData* d, int in_height, int in_width, int stride, int kernel_size, int in_channels, int out_channels) {
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

void _allocate_resnet_block_params(ResnetBlockParams* p, int embed_dim, int in_channels, int kernel_size, int time_embed_dim) {
	p->conv_1_kernels = malloc(embed_dim * sizeof(Matrix*));
	_allocate_conv_kernels(p->conv_1_kernels, kernel_size, in_channels, embed_dim);
	p->conv_2_kernels = malloc(embed_dim * sizeof(Matrix*));
	_allocate_conv_kernels(p->conv_2_kernels, kernel_size, embed_dim, embed_dim);
	p->residual_conv_kernels = malloc(embed_dim * sizeof(Matrix*));
	_allocate_conv_kernels(p->residual_conv_kernels, 1, in_channels, embed_dim);

	p->time_weights = malloc(sizeof(Matrix));
	p->time_weights->rows = time_embed_dim;
	p->time_weights->cols = embed_dim;
	p->time_weights->data = malloc(time_embed_dim * embed_dim * sizeof(matrix_float_t));

	p->time_biases = malloc(sizeof(Matrix));
	p->time_biases->rows = 1;
	p->time_biases->cols = embed_dim;
	p->time_biases->data = malloc(embed_dim * sizeof(matrix_float_t));
}

void _allocate_resnet_block_data(ResnetBlockData* d, int embed_dim, int height, int width, int in_channels, int kernel_size, int group_size) {
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
	_allocate_conv_data(d->conv_1, height, width, 1, kernel_size, in_channels, embed_dim);

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
	_allocate_conv_data(d->conv_2, height, width, 1, kernel_size, embed_dim, embed_dim);
	_allocate_conv_data(d->residual_conv, height, width, 1, 1, in_channels, embed_dim);
}

void _allocate_self_attention_block_params(SelfAttentionParams* p, int embed_dim, int key_dim) {
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
}

void _allocate_self_attention_block_data(SelfAttentionData* d, int embed_dim, int key_dim, int height, int width) {
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

	d->attention_weights_raw = malloc(sizeof(Matrix));
	d->attention_weights_raw->rows = spatial_dim;
	d->attention_weights_raw->cols = spatial_dim;
	d->attention_weights_raw->data = malloc(spatial_dim * spatial_dim * sizeof(matrix_float_t));

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

void allocate_model_params(ModelParams *p) {
	// First downsampling layer: resnet block x2, downsampling convolution
	p->down_1_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_1_resnet_1, RESOLUTION_1_EMBED_DIM, 3, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_1_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_1_conv_kernels = malloc(RESOLUTION_2_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->down_1_conv_kernels, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// First downsampling layer: resnet block, self-attention block, resnet block, self-attention block, downsampling convolution
	p->down_2_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_2_self_attention_1 = malloc(sizeof(SelfAttentionParams));
	_allocate_self_attention_block_params(p->down_2_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM);
	p->down_2_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_2_self_attention_2 = malloc(sizeof(SelfAttentionParams));
	_allocate_self_attention_block_params(p->down_2_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM);
	p->down_2_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->down_2_conv_kernels, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Third downsampling layer: resnet block x2 + downsampling convolution
	p->down_3_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_3_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_3_conv_kernels = malloc(RESOLUTION_4_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->down_3_conv_kernels, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM);

	// Fourth downsampling layer: resnet block x2
	p->down_4_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->down_4_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);

	// Mid layer: resnet block, self-attention block, resnet block
	p->mid_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->mid_self_attention = malloc(sizeof(SelfAttentionParams));
	_allocate_self_attention_block_params(p->mid_self_attention, RESOLUTION_4_EMBED_DIM, SELF_ATTENTION_KEY_DIM);
	p->mid_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);

	// First upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	p->up_1_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, 2 * RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_1_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_1_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->up_1_conv_kernels, KERNEL_SIZE, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Second upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	p->up_2_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, 2 * RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_2_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_2_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->up_2_conv_kernels, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// Third upsampling layer: resnet block, self-attention block, resnet block, self-attention block, nearest neighbours upsampling + a convolution
	p->up_3_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, 2 * RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_3_self_attention_1 = malloc(sizeof(SelfAttentionParams));
	_allocate_self_attention_block_params(p->up_3_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM);
	p->up_3_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_3_self_attention_2 = malloc(sizeof(SelfAttentionParams));
	_allocate_self_attention_block_params(p->up_3_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM);
	p->up_3_conv_kernels = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix*));
	_allocate_conv_kernels(p->up_3_conv_kernels, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);

	// Fourth upsampling layer: resnet block x2
	p->up_4_resnet_1 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, 2 * RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);
	p->up_4_resnet_2 = malloc(sizeof(ResnetBlockParams));
	_allocate_resnet_block_params(p->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, TIME_EMBED_DIM);

	// Output layer: group normalization, non-linearity, and a final convolution
	p->output_conv_kernels = malloc(3 * sizeof(Matrix*));
	_allocate_conv_kernels(p->output_conv_kernels, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, 3);
}

void allocate_model_data(ModelData* d) {
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
	d->down_1_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_1_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 3, KERNEL_SIZE, GROUP_SIZE);
	d->down_1_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_1_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->down_1_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// First downsampling layer: resnet block, self-attention block, resnet block, self-attention block, downsampling convolution
	d->down_2_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_2_self_attention_1 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block_data(d->down_2_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	d->down_2_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_2_self_attention_2 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block_data(d->down_2_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	d->down_2_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->down_2_conv, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Third downsampling layer: resnet block x2 + downsampling convolution
	d->down_3_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_3_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_3_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->down_3_conv, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESIZE_STRIDE, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM);

	// Fourth downsampling layer: resnet block x2
	d->down_4_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->down_4_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);

	// Mid layer: resnet block, self-attention block, resnet block
	d->mid_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->mid_self_attention = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block_data(d->mid_self_attention, RESOLUTION_4_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH);
	d->mid_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);

	// First upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	d->up_1_input_concat_skip = malloc(2 * RESOLUTION_4_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < 2 * RESOLUTION_4_EMBED_DIM; i++) {
		d->up_1_input_concat_skip[i].rows = RESOLUTION_4_HEIGHT;
		d->up_1_input_concat_skip[i].cols = RESOLUTION_4_WIDTH;
		d->up_1_input_concat_skip[i].data = malloc(RESOLUTION_4_HEIGHT * RESOLUTION_4_WIDTH * sizeof(matrix_float_t));
	}
	d->up_1_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, 2 * RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_1_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_1_nearest_neighbours = malloc(RESOLUTION_4_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_4_EMBED_DIM; i++) {
		d->up_1_nearest_neighbours[i].rows = RESOLUTION_3_HEIGHT;
		d->up_1_nearest_neighbours[i].cols = RESOLUTION_3_WIDTH;
		d->up_1_nearest_neighbours[i].data = malloc(RESOLUTION_3_HEIGHT * RESOLUTION_3_WIDTH * sizeof(matrix_float_t));
	}
	d->up_1_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->up_1_conv, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, 1, KERNEL_SIZE, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);

	// Second upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	d->up_2_input_concat_skip = malloc(2 * RESOLUTION_3_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < 2 * RESOLUTION_3_EMBED_DIM; i++) {
		d->up_2_input_concat_skip[i].rows = RESOLUTION_3_HEIGHT;
		d->up_2_input_concat_skip[i].cols = RESOLUTION_3_WIDTH;
		d->up_2_input_concat_skip[i].data = malloc(RESOLUTION_3_HEIGHT * RESOLUTION_3_WIDTH * sizeof(matrix_float_t));
	}
	d->up_2_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, 2 * RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_2_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_2_nearest_neighbours = malloc(RESOLUTION_3_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_3_EMBED_DIM; i++) {
		d->up_2_nearest_neighbours[i].rows = RESOLUTION_2_HEIGHT;
		d->up_2_nearest_neighbours[i].cols = RESOLUTION_2_WIDTH;
		d->up_2_nearest_neighbours[i].data = malloc(RESOLUTION_2_HEIGHT * RESOLUTION_2_WIDTH * sizeof(matrix_float_t));
	}
	d->up_2_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->up_2_conv, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, 1, KERNEL_SIZE, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);

	// Third upsampling layer: resnet block, self-attention block, resnet block, self-attention block, nearest neighbours upsampling + a convolution
	d->up_3_input_concat_skip = malloc(2 * RESOLUTION_2_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < 2 * RESOLUTION_2_EMBED_DIM; i++) {
		d->up_3_input_concat_skip[i].rows = RESOLUTION_2_HEIGHT;
		d->up_3_input_concat_skip[i].cols = RESOLUTION_2_WIDTH;
		d->up_3_input_concat_skip[i].data = malloc(RESOLUTION_2_HEIGHT * RESOLUTION_2_WIDTH * sizeof(matrix_float_t));
	}
	d->up_3_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, 2 * RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_3_self_attention_1 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block_data(d->up_3_self_attention_1, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	d->up_3_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_3_self_attention_2 = malloc(sizeof(SelfAttentionData));
	_allocate_self_attention_block_data(d->up_3_self_attention_2, RESOLUTION_2_EMBED_DIM, SELF_ATTENTION_KEY_DIM, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH);
	d->up_3_nearest_neighbours = malloc(RESOLUTION_2_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < RESOLUTION_2_EMBED_DIM; i++) {
		d->up_3_nearest_neighbours[i].rows = RESOLUTION_1_HEIGHT;
		d->up_3_nearest_neighbours[i].cols = RESOLUTION_1_WIDTH;
		d->up_3_nearest_neighbours[i].data = malloc(RESOLUTION_1_HEIGHT * RESOLUTION_1_WIDTH * sizeof(matrix_float_t));
	}
	d->up_3_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->up_3_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 1, KERNEL_SIZE, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);

	// Fourth upsampling layer: resnet block x2
	d->up_4_input_concat_skip = malloc(2 * RESOLUTION_1_EMBED_DIM * sizeof(Matrix));
	for (int i = 0; i < 2 * RESOLUTION_1_EMBED_DIM; i++) {
		d->up_4_input_concat_skip[i].rows = RESOLUTION_1_HEIGHT;
		d->up_4_input_concat_skip[i].cols = RESOLUTION_1_WIDTH;
		d->up_4_input_concat_skip[i].data = malloc(RESOLUTION_1_HEIGHT * RESOLUTION_1_WIDTH * sizeof(matrix_float_t));
	}
	d->up_4_resnet_1 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 2 * RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);
	d->up_4_resnet_2 = malloc(sizeof(ResnetBlockData));
	_allocate_resnet_block_data(d->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, KERNEL_SIZE, GROUP_SIZE);

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
	d->output_conv = malloc(sizeof(ConvData));
	_allocate_conv_data(d->output_conv, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 1, KERNEL_SIZE, RESOLUTION_1_EMBED_DIM, 3);
}

void _free_conv_kernels(Matrix** kernels, int in_channels, int out_channels) {
	for (int i = 0; i < out_channels; i++) {
		for (int j = 0; j < in_channels; j++) {
			free(kernels[i][j].data);
		}
		free(kernels[i]);
	}
}

void _free_conv_data(ConvData* d, int out_channels) {
	free(d->im2col->data);
	free(d->im2col);

	free(d->kernel_matrix->data);
	free(d->kernel_matrix);

	free(d->product->data);
	free(d->product);

	for (int i = 0; i < out_channels; i++) {
		free(d->output[i].data);
	}
	free(d->output);
}

void _free_resnet_block_params(ResnetBlockParams* p, int embed_dim, int in_channels) {
	_free_conv_kernels(p->conv_1_kernels, in_channels, embed_dim);
	free(p->conv_1_kernels);
	_free_conv_kernels(p->conv_2_kernels, embed_dim, embed_dim);
	free(p->conv_2_kernels);
	_free_conv_kernels(p->residual_conv_kernels, in_channels, embed_dim);
	free(p->residual_conv_kernels);

	free(p->time_weights->data);
	free(p->time_weights);

	free(p->time_biases->data);
	free(p->time_biases);
}

void _free_resnet_block_data(ResnetBlockData* d, int embed_dim, int in_channels) {
	free(d->group_norm_means_1);
	free(d->group_norm_stdevs_1);
	for (int i = 0; i < in_channels; i++) {
		free(d->relu_1[i].data);
	}
	free(d->relu_1);

	_free_conv_data(d->conv_1, embed_dim);
	free(d->conv_1);

	free(d->time_dense->data);
	free(d->time_dense);

	free(d->group_norm_means_2);
	free(d->group_norm_stdevs_2);

	// These three have the same dimensions, so just initialize them together
	for (int i = 0; i < embed_dim; i++) {
		free(d->relu_2[i].data);
		free(d->dropout[i].data);
		free(d->result[i].data);
	}
	free(d->relu_2);
	free(d->dropout);
	free(d->result);

	_free_conv_data(d->conv_2, embed_dim);
	_free_conv_data(d->residual_conv, embed_dim);
	free(d->conv_2);
	free(d->residual_conv);
}

void _free_self_attention_block_params(SelfAttentionParams* p) {
	free(p->Q_proj->data);
	free(p->Q_proj);

	free(p->K_proj->data);
	free(p->K_proj);

	free(p->V_proj->data);
	free(p->V_proj);

	free(p->weights->data);
	free(p->weights);

	free(p->biases->data);
	free(p->biases);
}

void _free_self_attention_block_data(SelfAttentionData* d, int embed_dim) {
	free(d->input->data);
	free(d->input);

	free(d->Q_proj->data);
	free(d->Q_proj);

	free(d->K_proj->data);
	free(d->K_proj);

	free(d->V_proj->data);
	free(d->V_proj);

	free(d->attention_weights->data);
	free(d->attention_weights);

	free(d->attention->data);
	free(d->attention);

	free(d->product->data);
	free(d->product);

	free(d->dense->data);
	free(d->dense);

	for (int i = 0; i < embed_dim; i++) {
		free(d->output[i].data);
	}
	free(d->output);
}

void free_model_params(ModelParams *p) {
	// First downsampling layer: resnet block x2, downsampling convolution
	_free_resnet_block_params(p->down_1_resnet_1, RESOLUTION_1_EMBED_DIM, 3);
	free(p->down_1_resnet_1);
	_free_resnet_block_params(p->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM);
	free(p->down_1_resnet_2);
	_free_conv_kernels(p->down_1_conv_kernels, RESOLUTION_1_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(p->down_1_conv_kernels);

	// First downsampling layer: resnet block, self-attention block, resnet block, self-attention block, downsampling convolution
	_free_resnet_block_params(p->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(p->down_2_resnet_1);
	_free_self_attention_block_params(p->down_2_self_attention_1);
	free(p->down_2_self_attention_1);
	_free_resnet_block_params(p->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(p->down_2_resnet_2);
	_free_self_attention_block_params(p->down_2_self_attention_2);
	free(p->down_2_self_attention_2);
	_free_conv_kernels(p->down_2_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(p->down_2_conv_kernels);

	// Third downsampling layer: resnet block x2 + downsampling convolution
	_free_resnet_block_params(p->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(p->down_3_resnet_1);
	_free_resnet_block_params(p->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(p->down_3_resnet_2);
	_free_conv_kernels(p->down_3_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->down_3_conv_kernels);

	// Fourth downsampling layer: resnet block x2
	_free_resnet_block_params(p->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->down_4_resnet_1);
	_free_resnet_block_params(p->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->down_4_resnet_2);

	// Mid layer: resnet block, self-attention block, resnet block
	_free_resnet_block_params(p->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->mid_resnet_1);
	_free_self_attention_block_params(p->mid_self_attention);
	free(p->mid_self_attention);
	_free_resnet_block_params(p->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->mid_resnet_2);

	// First upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	_free_resnet_block_params(p->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, 2 * RESOLUTION_4_EMBED_DIM);
	free(p->up_1_resnet_1);
	_free_resnet_block_params(p->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(p->up_1_resnet_2);
	_free_conv_kernels(p->up_1_conv_kernels, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(p->up_1_conv_kernels);

	// Second upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	_free_resnet_block_params(p->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, 2 * RESOLUTION_3_EMBED_DIM);
	free(p->up_2_resnet_1);
	_free_resnet_block_params(p->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(p->up_2_resnet_2);
	_free_conv_kernels(p->up_2_conv_kernels, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(p->up_2_conv_kernels);

	// Third upsampling layer: resnet block, self-attention block, resnet block, self-attention block, nearest neighbours upsampling + a convolution
	_free_resnet_block_params(p->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, 2 * RESOLUTION_2_EMBED_DIM);
	free(p->up_3_resnet_1);
	_free_self_attention_block_params(p->up_3_self_attention_1);
	free(p->up_3_self_attention_1);
	_free_resnet_block_params(p->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(p->up_3_resnet_2);
	_free_self_attention_block_params(p->up_3_self_attention_2);
	free(p->up_3_self_attention_2);
	_free_conv_kernels(p->up_3_conv_kernels, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);
	free(p->up_3_conv_kernels);

	// Fourth upsampling layer: resnet block x2
	_free_resnet_block_params(p->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, 2 * RESOLUTION_1_EMBED_DIM);
	free(p->up_4_resnet_1);
	_free_resnet_block_params(p->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM);
	free(p->up_4_resnet_2);

	// Output layer: group normalization, non-linearity, and a final convolution
	_free_conv_kernels(p->output_conv_kernels, RESOLUTION_1_EMBED_DIM, 3);
	free(p->output_conv_kernels);
}

void free_model_data(ModelData* d) {
	// Input data
	for (int i = 0; i < 3; i++) {
		free(d->X[i].data);
	}
	free(d->X);
	free(d->time_embedding->data);
	free(d->time_embedding);

	// First downsampling layer: resnet block x2, downsampling convolution
	_free_resnet_block_data(d->down_1_resnet_1, RESOLUTION_1_EMBED_DIM, 3);
	free(d->down_1_resnet_1);
	_free_resnet_block_data(d->down_1_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM);
	free(d->down_1_resnet_2);
	_free_conv_data(d->down_1_conv, RESOLUTION_2_EMBED_DIM);
	free(d->down_1_conv);

	// First downsampling layer: resnet block, self-attention block, resnet block, self-attention block, downsampling convolution
	_free_resnet_block_data(d->down_2_resnet_1, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(d->down_2_resnet_1);
	_free_self_attention_block_data(d->down_2_self_attention_1, RESOLUTION_2_EMBED_DIM);
	free(d->down_2_self_attention_1);
	_free_resnet_block_data(d->down_2_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(d->down_2_resnet_2);
	_free_self_attention_block_data(d->down_2_self_attention_2, RESOLUTION_2_EMBED_DIM);
	free(d->down_2_self_attention_2);
	_free_conv_data(d->down_2_conv, RESOLUTION_3_EMBED_DIM);
	free(d->down_2_conv);

	// Third downsampling layer: resnet block x2 + downsampling convolution
	_free_resnet_block_data(d->down_3_resnet_1, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(d->down_3_resnet_1);
	_free_resnet_block_data(d->down_3_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(d->down_3_resnet_2);
	_free_conv_data(d->down_3_conv, RESOLUTION_4_EMBED_DIM);
	free(d->down_3_conv);

	// Fourth downsampling layer: resnet block x2
	_free_resnet_block_data(d->down_4_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(d->down_4_resnet_1);
	_free_resnet_block_data(d->down_4_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(d->down_4_resnet_2);

	// Mid layer: resnet block, self-attention block, resnet block
	_free_resnet_block_data(d->mid_resnet_1, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(d->mid_resnet_1);
	_free_self_attention_block_data(d->mid_self_attention, RESOLUTION_4_EMBED_DIM);
	free(d->mid_self_attention);
	_free_resnet_block_data(d->mid_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(d->mid_resnet_2);

	// First upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	for (int i = 0; i < 2 * RESOLUTION_4_EMBED_DIM; i++) {
		free(d->up_1_input_concat_skip[i].data);
	}
	free(d->up_1_input_concat_skip);
	_free_resnet_block_data(d->up_1_resnet_1, RESOLUTION_4_EMBED_DIM, 2 * RESOLUTION_4_EMBED_DIM);
	free(d->up_1_resnet_1);
	_free_resnet_block_data(d->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM);
	free(d->up_1_resnet_2);
	for (int i = 0; i < RESOLUTION_4_EMBED_DIM; i++) {
		free(d->up_1_nearest_neighbours[i].data);
	}
	free(d->up_1_nearest_neighbours);
	_free_conv_data(d->up_1_conv, RESOLUTION_3_EMBED_DIM);
	free(d->up_1_conv);

	// Second upsampling layer: resnet block x2, nearest neighbours upsampling + a convolution
	for (int i = 0; i < 2 * RESOLUTION_3_EMBED_DIM; i++) {
		free(d->up_2_input_concat_skip[i].data);
	}
	free(d->up_2_input_concat_skip);
	_free_resnet_block_data(d->up_2_resnet_1, RESOLUTION_3_EMBED_DIM, 2 * RESOLUTION_3_EMBED_DIM);
	free(d->up_2_resnet_1);
	_free_resnet_block_data(d->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	free(d->up_2_resnet_2);
	for (int i = 0; i < RESOLUTION_3_EMBED_DIM; i++) {
		free(d->up_2_nearest_neighbours[i].data);
	}
	free(d->up_2_nearest_neighbours);
	_free_conv_data(d->up_2_conv, RESOLUTION_2_EMBED_DIM);
	free(d->up_2_conv);

	// Third upsampling layer: resnet block, self-attention block, resnet block, self-attention block, nearest neighbours upsampling + a convolution
	for (int i = 0; i < 2 * RESOLUTION_2_EMBED_DIM; i++) {
		free(d->up_3_input_concat_skip[i].data);
	}
	free(d->up_3_input_concat_skip);
	_free_resnet_block_data(d->up_3_resnet_1, RESOLUTION_2_EMBED_DIM, 2 * RESOLUTION_2_EMBED_DIM);
	free(d->up_3_resnet_1);
	_free_self_attention_block_data(d->up_3_self_attention_1, RESOLUTION_2_EMBED_DIM);
	free(d->up_3_self_attention_1);
	_free_resnet_block_data(d->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	free(d->up_3_resnet_2);
	_free_self_attention_block_data(d->up_3_self_attention_2, RESOLUTION_2_EMBED_DIM);
	free(d->up_3_self_attention_2);
	for (int i = 0; i < RESOLUTION_2_EMBED_DIM; i++) {
		free(d->up_3_nearest_neighbours[i].data);
	}
	free(d->up_3_nearest_neighbours);
	_free_conv_data(d->up_3_conv, RESOLUTION_1_EMBED_DIM);
	free(d->up_3_conv);

	// Fourth upsampling layer: resnet block x2
	for (int i = 0; i < 2 * RESOLUTION_1_EMBED_DIM; i++) {
		free(d->up_4_input_concat_skip[i].data);
	}
	free(d->up_4_input_concat_skip);
	_free_resnet_block_data(d->up_4_resnet_1, RESOLUTION_1_EMBED_DIM, 2 * RESOLUTION_1_EMBED_DIM);
	free(d->up_4_resnet_1);
	_free_resnet_block_data(d->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM);
	free(d->up_4_resnet_2);

	// Output layer: group normalization, non-linearity, and a final convolution
	free(d->output_group_norm_means);
	free(d->output_group_norm_stdevs);
	for (int i = 0; i < RESOLUTION_1_EMBED_DIM; i++) {
		free(d->output_relu[i].data);
	}
	free(d->output_relu);
	_free_conv_data(d->output_conv, 3);
	free(d->output_conv);
}

void _forward_attention(Matrix* X, SelfAttentionParams* params, SelfAttentionData* data) {
	// Computes unmasked scaled dot product attention
	int key_dimension = params->K_proj->cols;
	int spatial_dim = data->input->rows;
	reshape_channels_matrix(X, data->input);
	matrix_multiply_inplace(data->input, params->Q_proj, data->Q_proj);
	matrix_multiply_inplace(data->input, params->K_proj, data->K_proj);
	matrix_multiply_inplace(data->input, params->V_proj, data->V_proj);
	matrix_transpose(data->K_proj);
	matrix_multiply_inplace(data->Q_proj, data->K_proj, data->attention_weights);
	matrix_transpose(data->K_proj);
	matrix_scale(data->attention_weights, 1.0 / sqrt(key_dimension));
	// Make a copy before softmax for use in backward pass
	for (int i = 0; i < spatial_dim * spatial_dim; i++) {
		data->attention_weights_raw->data[i] = data->attention_weights->data[i];
	}
	softmax_row_wise(data->attention_weights->data, data->attention_weights->rows, data->attention_weights->cols);

	// Project attention back to (height * width, channels), then reshape to (channels, height, width)
	matrix_multiply_inplace(data->attention_weights, data->V_proj, data->attention);
	matrix_multiply_inplace(data->attention, params->weights, data->dense);
	matrix_add_tile_rows(data->dense, params->biases);
	reshape_matrix_channels(data->dense, data->output);
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
	_add_time_embedding(d->conv_1->output, d->time_dense, out_channels);

	// Second chunk
	group_norm(d->conv_1->output, d->relu_2, d->group_norm_stdevs_2, d->group_norm_means_2, out_channels, group_size);
	multi_channel_relu(d->relu_2, out_channels);
	_dropout(d->relu_2, d->dropout, out_channels);
	conv(d->dropout, p->conv_2_kernels, d->conv_2, out_channels, out_channels, 1);

	// Residual skip connection
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

void _concat_skip(Matrix* in, Matrix* skip, Matrix* out, int channels) {
	int height = out[0].rows;
	int width = out[0].cols;
	for (int c = 0; c < channels; c++) {
		memcpy(out[c].data, in[c].data, sizeof(matrix_float_t) * height * width);
	}
	for (int c = 0; c < channels; c++) {
		memcpy(out[c + channels].data, skip[c].data, sizeof(matrix_float_t) * height * width);
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
	_concat_skip(d->mid_resnet_2->result, d->down_4_resnet_2->result, d->up_1_input_concat_skip, RESOLUTION_4_EMBED_DIM);
	_forward_resnet(d->up_1_input_concat_skip, d->time_embedding, p->up_1_resnet_1, d->up_1_resnet_1, 2 * RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_1_resnet_1->result, d->time_embedding, p->up_1_resnet_2, d->up_1_resnet_2, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_nearest_neighbours(d->up_1_resnet_2->result, d->up_1_nearest_neighbours, RESOLUTION_4_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_1_nearest_neighbours;
	if (RESOLUTION_4_EMBED_DIM != RESOLUTION_3_EMBED_DIM) {
		conv(d->up_1_nearest_neighbours, p->up_1_conv_kernels, d->up_1_conv, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM, 1);
		next = d->up_1_conv->output;
	}
	
	_concat_skip(next, d->down_3_resnet_2->result, d->up_2_input_concat_skip, RESOLUTION_3_EMBED_DIM);
	_forward_resnet(d->up_2_input_concat_skip, d->time_embedding, p->up_2_resnet_1, d->up_2_resnet_1, 2 * RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_2_resnet_1->result, d->time_embedding, p->up_2_resnet_2, d->up_2_resnet_2, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_nearest_neighbours(d->up_2_resnet_2->result, d->up_2_nearest_neighbours, RESOLUTION_3_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_2_nearest_neighbours;
	if (RESOLUTION_3_EMBED_DIM != RESOLUTION_2_EMBED_DIM) {
		conv(d->up_2_nearest_neighbours, p->up_2_conv_kernels, d->up_2_conv, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM, 1);
		next = d->up_2_conv->output;
	}
	
	_concat_skip(next, d->down_2_resnet_2->result, d->up_3_input_concat_skip, RESOLUTION_2_EMBED_DIM);
	_forward_resnet(d->up_3_input_concat_skip, d->time_embedding, p->up_3_resnet_1, d->up_3_resnet_1, 2 * RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->up_3_resnet_1->result, p->up_3_self_attention_1, d->up_3_self_attention_1);
	_forward_resnet(d->up_3_self_attention_1->output, d->time_embedding, p->up_3_resnet_2, d->up_3_resnet_2, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_forward_attention(d->up_3_resnet_2->result, p->up_3_self_attention_1, d->up_3_self_attention_1);
	_nearest_neighbours(d->up_3_self_attention_2->output, d->up_3_nearest_neighbours, RESOLUTION_2_EMBED_DIM, RESIZE_STRIDE);
	next = d->up_3_nearest_neighbours;
	if (RESOLUTION_2_EMBED_DIM != RESOLUTION_1_EMBED_DIM) {
		conv(d->up_3_nearest_neighbours, p->up_3_conv_kernels, d->up_3_conv, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM, 1);
		next = d->up_3_conv->output;
	}
	
	_concat_skip(next, d->down_1_resnet_2->result, d->up_4_input_concat_skip, RESOLUTION_1_EMBED_DIM);
	_forward_resnet(d->up_4_input_concat_skip, d->time_embedding, p->up_4_resnet_1, d->up_4_resnet_1, 2 * RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	_forward_resnet(d->up_4_resnet_1->result, d->time_embedding, p->up_4_resnet_2, d->up_4_resnet_2, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);

	// Output
	group_norm(d->up_4_resnet_2->result, d->output_relu, d->output_group_norm_stdevs, d->output_group_norm_means, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	multi_channel_relu(d->output_relu, RESOLUTION_1_EMBED_DIM);
	conv(d->output_relu, p->output_conv_kernels, d->output_conv, RESOLUTION_1_EMBED_DIM, 3, 1);
}

void _dropout_mask(Matrix* X, Matrix* dropout_result, int channels) {
	int height = X[0].rows;
	int width = X[0].cols;
	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < height * width; i++) {
			if (dropout_result[c].data[i] == 0) {
				X[c].data[i] = 0;
			}
		}
	}
}

void _backward_resnet(ResnetBlockParams* p, ResnetBlockParams* g, ResnetBlockData* d, ResnetBlockData* gd, Matrix* del_input, Matrix* input, Matrix* time_emb, int in_channels, int out_channels, int group_size) {
	int height = input[0].rows;
	int width = input[0].cols;
	Matrix* del_output = gd->result;
	
	// Gradient w.r.t. second chunk
	conv_ddx(del_output, d->conv_2, gd->conv_2, g->conv_2_kernels, gd->dropout, out_channels, 1);
	_dropout_mask(gd->dropout, d->dropout, out_channels);
	multi_channel_relu_ddx(gd->dropout, gd->relu_2, d->relu_2, out_channels);
	group_norm_ddx(gd->relu_2, gd->conv_1->output, d->conv_1->output, d->group_norm_means_2, d->group_norm_stdevs_2, out_channels, group_size);
	
	// Gradient w.r.t. time embedding projection parameters
	for (int c = 0; c < out_channels; c++) {
			g->time_biases->data[c] = 0;
		for (int i = 0; i < height * width; i++) {
			g->time_biases->data[c] += gd->conv_1->output[c].data[i];
		}
	}
	matrix_transpose(time_emb);
	matrix_multiply_inplace(time_emb, g->time_biases, g->time_weights);
	matrix_transpose(time_emb);

	// Gradient w.r.t. first chunk
	conv_ddx(gd->conv_1->output, d->conv_1, gd->conv_1, p->conv_1_kernels, gd->relu_1, in_channels, 1);
	multi_channel_relu_ddx(gd->relu_1, gd->relu_1, d->relu_1, in_channels);
	group_norm_ddx(gd->relu_1, del_input, input, d->group_norm_means_1, d->group_norm_stdevs_1, in_channels, group_size);
	
	// Residual skip connection
	Matrix* residual_grad_connection = del_output;
	if (in_channels != out_channels) { // TODO: Tidy this up to avoid malloc and free
		residual_grad_connection = malloc(in_channels * sizeof(Matrix));
		for (int c = 0; c < in_channels; c++) {
			residual_grad_connection[c].rows = height;
			residual_grad_connection[c].cols = width;
			residual_grad_connection[c].data = malloc(height * width * sizeof(matrix_float_t));
		}
		conv_ddx(del_output, d->residual_conv, gd->residual_conv, p->residual_conv_kernels, residual_grad_connection, in_channels, 1);
	}
	for (int c = 0; c < in_channels; c++) {
		matrix_add(&del_input[c], &residual_grad_connection[c]);
	}
	if (in_channels != out_channels) { // TODO: Tidy this up to avoid malloc and free
		for (int c = 0; c < in_channels; c++) {
			free(residual_grad_connection[c].data);
		}
		free(residual_grad_connection);
	}
}

void _nearest_neighbours_ddx(Matrix* source, Matrix* dest, int channels, int scale) {
	int source_height = source[0].rows;
	int source_width = source[0].cols;
	int dest_height = dest[0].rows;
	int dest_width = dest[0].cols;

	for (int c = 0; c < channels; c++) {
		memset(dest[c].data, 0, dest_height * dest_width * sizeof(matrix_float_t));
		
		for (int i = 0; i < source_height; i++) {
			for (int j = 0; j < source_width; j++) {
				dest[c].data[(i / scale) * dest_width + j / scale] += source[c].data[i * source_width + j];
			}
		}
	}
}

void _softmax_ddx(Matrix* softmax_output, Matrix* gradient, Matrix* output) {
	// Compute the derivative w.r.t. softmax as the row-wise product of the softmax Jacobian with the incoming gradient
	int dim = softmax_output->cols;
	for (int i = 0; i < dim; i++) {
		matrix_float_t dot = 0;
		for (int j = 0; j < dim; j++) {
			dot += softmax_output->data[i * dim + j] * gradient->data[i * dim + j];
		}
		
		for (int j = 0; j < dim; j++) {
			output->data[i * dim + j] = softmax_output->data[i * dim + j] * (gradient->data[i * dim + j] - dot);
		}
	}
}

void _backward_attention(SelfAttentionParams* p, SelfAttentionData* d, SelfAttentionParams* g, SelfAttentionData* gd, Matrix* del_input, int key_dim) {
	// Aliasing different objects in memory so they match the variables I have in my written work (easier to find errors)
	Matrix* del_Y = gd->output;
	Matrix* del_Y_prime = gd->product;
	Matrix* P = d->attention;
	Matrix* del_P = gd->attention;
	Matrix* S = d->attention_weights;
	Matrix* del_S = gd->attention_weights;
	Matrix* W = p->weights;
	Matrix* del_W = g->weights;
	Matrix* K = d->K_proj;
	Matrix* del_K = gd->K_proj;
	Matrix* K_proj = p->K_proj;
	Matrix* del_K_proj = g->K_proj;
	Matrix* Q = d->Q_proj;
	Matrix* del_Q = gd->Q_proj;
	Matrix* Q_proj = p->Q_proj;
	Matrix* del_Q_proj = g->Q_proj;
	Matrix* V = d->V_proj;
	Matrix* del_V = gd->V_proj;
	Matrix* V_proj = p->V_proj;
	Matrix* del_V_proj = g->V_proj;
	Matrix* Z = d->input;
	Matrix* del_Z = gd->input;
	Matrix* del_Z2 = d->product; // Use the matrix field in d since it is unused and we use the field in gd already
	Matrix* I = d->attention_weights_raw;
	Matrix* del_I = gd->attention_weights_raw;

	reshape_channels_matrix(del_Y, del_Y_prime);

	matrix_transpose(P);
	matrix_multiply_inplace(P, del_Y_prime, del_W);
	matrix_transpose(P);

	matrix_transpose(W);
	matrix_multiply_inplace(del_Y_prime, W, del_P);
	matrix_transpose(W);

	matrix_transpose(S);
	matrix_multiply_inplace(S, del_P, del_V);
	matrix_transpose(S);

	matrix_transpose(V);
	matrix_multiply_inplace(del_P, V, del_S);
	matrix_transpose(V);

	_softmax_ddx(I, del_S, del_I);
	matrix_scale(del_I, 1.0 / sqrt(key_dim));

	matrix_multiply_inplace(del_I, K, del_Q);
	
	matrix_transpose(del_I);
	matrix_multiply_inplace(del_I, Q, del_K);
	matrix_transpose(del_I);

	matrix_transpose(Z);
	matrix_multiply_inplace(Z, del_K, del_K_proj);
	matrix_multiply_inplace(Z, del_Q, del_Q_proj);
	matrix_multiply_inplace(Z, del_V, del_V_proj);
	matrix_transpose(Z);

	matrix_transpose(Q_proj);
	matrix_multiply_inplace(del_Q, Q_proj, del_Z);
	matrix_transpose(Q_proj);
	matrix_transpose(K_proj);
	matrix_multiply_inplace(del_K, K_proj, del_Z2);
	matrix_add(del_Z, del_Z2);
	matrix_transpose(K_proj);
	matrix_transpose(V_proj);
	matrix_multiply_inplace(del_V, V_proj, del_Z2);
	matrix_add(del_Z, del_Z2);
	matrix_transpose(V_proj);
	
	reshape_matrix_channels(del_Z, del_input);
}

typedef uint8_t bool;
// Assumes that src has 2 * dest_channels channels and dest has dest_channels
void _split_concat(Matrix* src, Matrix* dest, int dest_channels, bool second_half) {
	int height = dest[0].rows;
	int width = dest[0].cols;
	int index_offset = 0;
	if (second_half == 1) {
		index_offset = dest_channels;
	}
	for (int c = 0; c < dest_channels; c++) {
		memcpy(&dest[c], &src[c + index_offset], height * width * sizeof(matrix_float_t));
	}
}

void backward(ModelParams* p, ModelData* d, ModelParams* g, ModelData* gd, Matrix* noise) {
	// Deriv. of loss w.r.t. predicted output
	Matrix* del_Y = gd->output_conv->output;
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < RESOLUTION_1_HEIGHT; i++) {
			for (int j = 0; j < RESOLUTION_1_WIDTH; j++) {
				gd->output_conv->output[c].data[i * RESOLUTION_1_WIDTH + j] = d->output_conv->output[c].data[i * RESOLUTION_1_WIDTH + j];
			}
		}
		matrix_scale(&noise[c], -1);
		matrix_add(&gd->output_conv->output[c], &noise[c]);
		matrix_scale(&noise[c], -1);
		matrix_scale(&gd->output_conv->output[c], 2);
	}

	// Output processing
	conv_ddx(del_Y, d->output_conv, gd->output_conv, g->output_conv_kernels, gd->output_relu, RESOLUTION_1_EMBED_DIM, 1);
	multi_channel_relu_ddx(gd->output_relu, gd->output_relu, d->output_relu, RESOLUTION_1_EMBED_DIM);
	group_norm_ddx(gd->output_relu, gd->up_4_resnet_2->result, d->up_4_resnet_2->result, d->output_group_norm_means, d->output_group_norm_stdevs, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);

	// Fourth upsampling layer
	_backward_resnet(p->up_4_resnet_2, g->up_4_resnet_2, d->up_4_resnet_2, gd->up_4_resnet_2, gd->up_4_resnet_1->result, d->up_4_resnet_1->result, d->time_embedding, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->up_4_resnet_1, g->up_4_resnet_1, d->up_4_resnet_1, gd->up_4_resnet_1, gd->up_4_input_concat_skip, d->up_4_input_concat_skip, d->time_embedding, 2 * RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
	_split_concat(gd->up_4_input_concat_skip, gd->up_3_conv->output, RESOLUTION_1_EMBED_DIM, 0);

	// Third upsampling layer
	conv_ddx(gd->up_3_conv->output, d->up_3_conv, gd->up_3_conv, p->up_3_conv_kernels, gd->up_3_nearest_neighbours, RESOLUTION_2_EMBED_DIM, 1);
	_nearest_neighbours_ddx(gd->up_3_nearest_neighbours, gd->up_3_self_attention_2->output, RESOLUTION_2_EMBED_DIM, RESIZE_STRIDE);
	_backward_attention(p->up_3_self_attention_2, d->up_3_self_attention_2, g->up_3_self_attention_2, gd->up_3_self_attention_2, gd->up_3_resnet_2->result, SELF_ATTENTION_KEY_DIM);
	_backward_resnet(p->up_3_resnet_2, g->up_3_resnet_2, d->up_3_resnet_2, gd->up_3_resnet_2, gd->up_3_self_attention_1->output, d->up_3_self_attention_1->output, d->time_embedding, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_backward_attention(p->up_3_self_attention_1, d->up_3_self_attention_1, g->up_3_self_attention_1, gd->up_3_self_attention_1, gd->up_3_resnet_1->result, SELF_ATTENTION_KEY_DIM);
	_backward_resnet(p->up_3_resnet_1, g->up_3_resnet_1, d->up_3_resnet_1, gd->up_3_resnet_1, gd->up_3_input_concat_skip, d->up_3_input_concat_skip, d->time_embedding, 2 * RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	_split_concat(gd->up_3_input_concat_skip, gd->up_2_conv->output, RESOLUTION_2_EMBED_DIM, 0);

	// Second upsampling layer
	conv_ddx(gd->up_2_conv->output, d->up_2_conv, gd->up_2_conv, p->up_2_conv_kernels, gd->up_2_nearest_neighbours, RESOLUTION_3_EMBED_DIM, 1);
	_nearest_neighbours_ddx(gd->up_2_nearest_neighbours, gd->up_2_resnet_2->result, RESOLUTION_3_EMBED_DIM, RESIZE_STRIDE);
	_backward_resnet(p->up_2_resnet_2, g->up_2_resnet_2, d->up_2_resnet_2, gd->up_2_resnet_2, gd->up_2_resnet_1->result, d->up_2_resnet_1->result, d->time_embedding, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->up_2_resnet_1, g->up_2_resnet_1, d->up_2_resnet_1, gd->up_2_resnet_1, gd->up_2_input_concat_skip, d->up_2_input_concat_skip, d->time_embedding, 2 * RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_split_concat(gd->up_2_input_concat_skip, gd->up_1_conv->output, RESOLUTION_3_EMBED_DIM, 0);

	// First upsampling layer
	conv_ddx(gd->up_1_conv->output, d->up_1_conv, gd->up_1_conv, p->up_1_conv_kernels, gd->up_1_nearest_neighbours, RESOLUTION_4_EMBED_DIM, 1);
	_nearest_neighbours_ddx(gd->up_1_nearest_neighbours, gd->up_1_resnet_2->result, RESOLUTION_4_EMBED_DIM, RESIZE_STRIDE);
	_backward_resnet(p->up_1_resnet_2, g->up_1_resnet_2, d->up_1_resnet_2, gd->up_1_resnet_2, gd->up_1_resnet_1->result, d->up_1_resnet_1->result, d->time_embedding, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->up_1_resnet_1, g->up_1_resnet_1, d->up_1_resnet_1, gd->up_1_resnet_1, gd->up_1_input_concat_skip, d->up_1_input_concat_skip, d->time_embedding, 2 * RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_split_concat(gd->up_1_input_concat_skip, gd->mid_resnet_2->result, RESOLUTION_4_EMBED_DIM, 0);

	// Mid layer
	_backward_resnet(p->mid_resnet_2, g->mid_resnet_2, d->mid_resnet_2, gd->mid_resnet_2, gd->mid_self_attention->output, d->mid_self_attention->output, d->time_embedding, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_backward_attention(p->mid_self_attention, d->mid_self_attention, g->mid_self_attention, gd->mid_self_attention, gd->mid_resnet_1->result, SELF_ATTENTION_KEY_DIM);
	_backward_resnet(p->mid_resnet_1, g->mid_resnet_1, d->mid_resnet_1, gd->mid_resnet_1, gd->down_4_resnet_2->result, d->down_4_resnet_2->result, d->time_embedding, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);

	// Fourth downsampling layer
	for (int c = 0; c < RESOLUTION_4_EMBED_DIM; c++) {
		matrix_add(&gd->down_4_resnet_2->result[c], &gd->up_1_input_concat_skip[c + RESOLUTION_4_EMBED_DIM]);
	}
	_backward_resnet(p->down_4_resnet_2, g->down_4_resnet_2, d->down_4_resnet_2, gd->down_4_resnet_2, gd->down_4_resnet_1->result, d->down_4_resnet_1->result, d->time_embedding, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->down_4_resnet_1, g->down_4_resnet_1, d->down_4_resnet_1, gd->down_4_resnet_1, gd->down_3_conv->output, d->down_3_conv->output, d->time_embedding, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, GROUP_SIZE);
	
	// Third downsampling layer
	conv_ddx(gd->down_3_conv->output, d->down_3_conv, gd->down_3_conv, p->down_3_conv_kernels, gd->down_3_resnet_2->result, RESOLUTION_3_EMBED_DIM, 1);
	for (int c = 0; c < RESOLUTION_3_EMBED_DIM; c++) {
		matrix_add(&gd->down_3_resnet_2->result[c], &gd->up_2_input_concat_skip[c + RESOLUTION_3_EMBED_DIM]);
	}
	_backward_resnet(p->down_3_resnet_2, g->down_3_resnet_2, d->down_3_resnet_2, gd->down_3_resnet_2, gd->down_3_resnet_1->result, d->down_3_resnet_1->result, d->time_embedding, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->down_3_resnet_1, g->down_3_resnet_1, d->down_3_resnet_1, gd->down_3_resnet_1, gd->down_2_conv->output, d->down_2_conv->output, d->time_embedding, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	
	// Second downsampling layer
	conv_ddx(gd->down_2_conv->output, d->down_2_conv, gd->down_2_conv, p->down_2_conv_kernels, gd->down_2_self_attention_2->output, RESOLUTION_2_EMBED_DIM, 1);
	for (int c = 0; c < RESOLUTION_2_EMBED_DIM; c++) {
		matrix_add(&gd->down_2_resnet_2->result[c], &gd->up_3_input_concat_skip[c + RESOLUTION_2_EMBED_DIM]);
	}
	_backward_attention(p->down_2_self_attention_2, d->down_2_self_attention_2, g->down_2_self_attention_2, gd->down_2_self_attention_2, gd->down_2_resnet_2->result, SELF_ATTENTION_KEY_DIM);
	_backward_resnet(p->down_2_resnet_2, g->down_2_resnet_2, d->down_2_resnet_2, gd->down_2_resnet_2, gd->down_2_self_attention_2->output, d->down_2_self_attention_2->output, d->time_embedding, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_backward_attention(p->down_2_self_attention_1, d->down_2_self_attention_1, g->down_2_self_attention_1, gd->down_2_self_attention_1, gd->down_2_resnet_1->result, SELF_ATTENTION_KEY_DIM);
	_backward_resnet(p->down_2_resnet_1, g->down_2_resnet_1, d->down_2_resnet_1, gd->down_2_resnet_1, gd->down_1_conv->output, d->down_1_conv->output, d->time_embedding, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, GROUP_SIZE);
	
	// First downsampling layer
	conv_ddx(gd->down_1_conv->output, d->down_1_conv, gd->down_1_conv, p->down_1_conv_kernels, gd->down_1_resnet_2->result, RESOLUTION_1_EMBED_DIM, 1);
	for (int c = 0; c < RESOLUTION_1_EMBED_DIM; c++) {
		matrix_add(&gd->down_1_resnet_2->result[c], &gd->up_4_input_concat_skip[c + RESOLUTION_1_EMBED_DIM]);
	}
	_backward_resnet(p->down_1_resnet_2, g->down_1_resnet_2, d->down_1_resnet_2, gd->down_1_resnet_2, gd->down_1_resnet_1->result, d->down_3_resnet_1->result, d->time_embedding, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, GROUP_SIZE);
	_backward_resnet(p->down_1_resnet_1, g->down_1_resnet_1, d->down_1_resnet_1, gd->down_1_resnet_1, gd->X, d->X, d->time_embedding, 3, RESOLUTION_1_EMBED_DIM, GROUP_SIZE);
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
	
	_init_resnet_block(p->up_1_resnet_1, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, 2 * RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_1_resnet_2, RESOLUTION_4_HEIGHT, RESOLUTION_4_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_4_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->up_1_conv_kernels, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_4_EMBED_DIM, RESOLUTION_3_EMBED_DIM);
	
	_init_resnet_block(p->up_2_resnet_1, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, 2 * RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_2_resnet_2, RESOLUTION_3_HEIGHT, RESOLUTION_3_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_3_EMBED_DIM, TIME_EMBED_DIM);
	_init_conv_kernels(p->up_2_conv_kernels, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_3_EMBED_DIM, RESOLUTION_2_EMBED_DIM);
	
	_init_resnet_block(p->up_3_resnet_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, 2 * RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->up_3_self_attention_1, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_resnet_block(p->up_3_resnet_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_2_EMBED_DIM, TIME_EMBED_DIM);
	_init_self_attention_block(p->up_3_self_attention_2, RESOLUTION_2_HEIGHT, RESOLUTION_2_WIDTH, SELF_ATTENTION_KEY_DIM);
	_init_conv_kernels(p->up_3_conv_kernels, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_2_EMBED_DIM, RESOLUTION_1_EMBED_DIM);

	_init_resnet_block(p->up_4_resnet_1, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, 2 * RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);
	_init_resnet_block(p->up_4_resnet_2, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, RESOLUTION_1_EMBED_DIM, TIME_EMBED_DIM);

	_init_conv_kernels(p->output_conv_kernels, RESOLUTION_1_HEIGHT, RESOLUTION_1_WIDTH, RESOLUTION_1_EMBED_DIM, 3);
}

void init() {
	ModelParams p;

	allocate_model_params(&p);

	init_parameters(&p);

	save_parameters(&p);

	free_model_params(&p);
}

float compute_mse_loss(Matrix* expected, Matrix* actual, int channels) {
	float loss = 0;
	int height = expected[0].rows;
	int width = expected[0].cols;
	for (int c = 0; c < channels; c++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float residual = actual[c].data[i * width + j] - expected[c].data[i * width + j];
				loss += residual * residual;
			}
		}
	}
	loss /= channels * height * width;
	return loss;
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
	ModelParams g; // Gradient
	ModelParams gm; // Gradient moment
	ModelParams gsm; // Gradient squared moment
	ModelData gd; // Gradient data (for intermittent calculations)

	allocate_model_params(&p);
	allocate_model_data(&d);
	allocate_model_params(&g);
	allocate_model_params(&gm);
	allocate_model_params(&gsm);
	allocate_model_data(&gd);

	// Speed up testing by avoiding file system operations
	// load_parameters(&p);
	init_parameters(&p);

	load_example(d.X, data_fds[0]);
	Matrix noise[3]; // TODO: Generate random noise (and malloc data to store it in)
	forward(&p, &d);
	float loss = compute_mse_loss(noise, d.output_conv->output, 3);
	(void) loss;
	backward(&p, &d, &g, &gd, noise);

	for (int i = 0; i < 5; i++) {
		close(data_fds[i]);
	}

	free_model_params(&p);
	free_model_data(&d);
	free_model_params(&g);
	free_model_params(&gm);
	free_model_params(&gsm);
	free_model_data(&gd);
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