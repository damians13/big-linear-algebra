// MNIST classification using a neural entwork
#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/mnist_csv2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define SGD_BATCH_SIZE 64
#define SGD_LEARN_RATE_MULTIPLIER 0.02
#define SGD_GRADIENT_CLIP INFINITY // Set this to a finite value to clip gradients past this threshold

#define LOSS_EPSILON 1e-15 // Avoid log(0)

/**
 * Model architecture:
 * - Input
 * - 1st hidden layer (ReLU)
 * - 2nd hidden layer (ReLU)
 * - Output layer (Softmax)
 */

#define LAYER_INPUT_SIZE 784
#define LAYER_1_SIZE 256
#define LAYER_2_SIZE 128
#define LAYER_3_SIZE 10

#define LAYER_1_WEIGHT_FILE "data/mnist_nn/weights_1.csv"
#define LAYER_2_WEIGHT_FILE "data/mnist_nn/weights_2.csv"
#define LAYER_3_WEIGHT_FILE "data/mnist_nn/weights_3.csv"
#define LAYER_1_BIAS_FILE "data/mnist_nn/biases_1.csv"
#define LAYER_2_BIAS_FILE "data/mnist_nn/biases_2.csv"
#define LAYER_3_BIAS_FILE "data/mnist_nn/biases_3.csv"

// For forward pass
void relu(matrix_float_t* data, int num) {
	for (int i = 0; i < num; i++) {
		if (data[i] < 0) {
			data[i] = 0;
		}
	}
}

// For backpropagation
void relu_ddx(matrix_float_t* data, int num) {
	for (int i = 0; i < num; i++) {
		data[i] = data[i] > 0 ? 1 : 0;
	}
}

// To convert to probabilities in final layer. Applied individually per column
void softmax(matrix_float_t* data, int rows, int cols) {
	for (int i = 0; i < cols; i++) {
		// Subtract the max value from each value to avoid overflow in expf
		matrix_float_t max_value = -INFINITY;
		for (int j = 0; j < rows; j++) {
			if (data[i + j * cols] > max_value) {
				max_value = data[i + j * cols];
			}
		}

		matrix_float_t sum_of_exponents = 0;
		for (int j = 0; j < rows; j++) {
			data[i + j * cols] = exp(data[i + j * cols] - max_value);
			sum_of_exponents += data[i + j * cols];
		}
		for (int j = 0; j < rows; j++) {
			data[i + j * cols] /= sum_of_exponents;
		}
	}
}

// To help prevent exploding gradients (not used currently)
void clip_gradient(Matrix* g) {
	matrix_float_t norm = frobenius_norm(*g);
	if (norm > SGD_GRADIENT_CLIP) {
		matrix_scale(g, SGD_GRADIENT_CLIP / norm);
	}
}

double cross_entropy_loss(matrix_float_t* actual_dist, matrix_float_t* expected_dist, int size) {
	matrix_float_t loss = 0;
	for (int i = 0; i < size; i++) {
		matrix_float_t exp_value = actual_dist[i];
		matrix_float_t val = -1 * (expected_dist[i] * log(exp_value + LOSS_EPSILON));
		loss += val;
	}
	return loss;
}

/**
 * Initialize model parameters by sampling from a uniform distribution (since it's easier than a Gaussian) using He initialization
 * Biases are initialized to 0
 */
void init() {
	// 1st hidden layer weights
	float weight_data_1[LAYER_INPUT_SIZE * LAYER_1_SIZE];
	float layer_1_range = 2 * sqrtf(6.0 / (float) LAYER_INPUT_SIZE);
	for (int i = 0; i < LAYER_INPUT_SIZE * LAYER_1_SIZE; i++) {
		weight_data_1[i] = layer_1_range * (float)rand()/(float)(RAND_MAX) - layer_1_range / 2;
	}
	write_csv_contents(LAYER_1_WEIGHT_FILE, weight_data_1, LAYER_INPUT_SIZE, LAYER_1_SIZE);

	// 2nd hidden layer weights
	float weight_data_2[LAYER_1_SIZE * LAYER_2_SIZE];
	float layer_2_range = 2 * sqrtf(6.0 / (float) LAYER_1_SIZE);
	for (int i = 0; i < LAYER_1_SIZE * LAYER_2_SIZE; i++) {
		weight_data_2[i] = layer_2_range * (float)rand()/(float)(RAND_MAX) - layer_2_range / 2;
	}
	write_csv_contents(LAYER_2_WEIGHT_FILE, weight_data_2, LAYER_1_SIZE, LAYER_2_SIZE);

	// Output layer weights
	float weight_data_3[LAYER_2_SIZE * LAYER_3_SIZE];
	float layer_3_range = 2 * sqrtf(6.0 / (float) LAYER_2_SIZE);
	for (int i = 0; i < LAYER_2_SIZE * LAYER_3_SIZE; i++) {
		weight_data_3[i] = layer_3_range * (float)rand()/(float)(RAND_MAX) - layer_3_range / 2;
	}
	write_csv_contents(LAYER_3_WEIGHT_FILE, weight_data_3, LAYER_2_SIZE, LAYER_3_SIZE);

	// 1st hidden layer biases
	float bias_data_1[LAYER_1_SIZE];
	for (int i = 0; i < LAYER_1_SIZE; i++) {
		bias_data_1[i] = 0;
	}
	write_csv_contents(LAYER_1_BIAS_FILE, bias_data_1, 1, LAYER_1_SIZE);

	// 2nd hidden layer biases
	float bias_data_2[LAYER_2_SIZE];
	for (int i = 0; i < LAYER_2_SIZE; i++) {
		bias_data_2[i] = 0;
	}
	write_csv_contents(LAYER_2_BIAS_FILE, bias_data_2, 1, LAYER_2_SIZE);

	// Output layer biases
	float bias_data_3[LAYER_3_SIZE];
	for (int i = 0; i < LAYER_3_SIZE; i++) {
		bias_data_3[i] = 0;
	}
	write_csv_contents(LAYER_3_BIAS_FILE, bias_data_3, 1, LAYER_3_SIZE);
}

/**
 * 
 */
struct Matrix* load_matrix_from_csv(const char* filepath, int rows, int cols) {
	struct Matrix* matrix = malloc(sizeof(struct Matrix));
	float* csv_data = read_csv_contents(filepath);
	matrix_float_t* matrix_data = malloc(rows * cols * sizeof(matrix_float_t));
	for (int i = 0; i < rows * cols; i++) {
		matrix_data[i] = (matrix_float_t) csv_data[i];
	}
	free(csv_data);
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->data = matrix_data;
	return matrix;
}

/**
 * Train the model for `num_epochs` epochs via stochastic gradient descent
 */
void train(int num_epochs) {
	Matrix* layer_1_weights = load_matrix_from_csv(LAYER_1_WEIGHT_FILE, LAYER_1_SIZE, LAYER_INPUT_SIZE);
	Matrix* layer_2_weights = load_matrix_from_csv(LAYER_2_WEIGHT_FILE, LAYER_2_SIZE, LAYER_1_SIZE);
	Matrix* layer_3_weights = load_matrix_from_csv(LAYER_3_WEIGHT_FILE, LAYER_3_SIZE, LAYER_2_SIZE);
	Matrix* layer_1_biases = load_matrix_from_csv(LAYER_1_BIAS_FILE, LAYER_1_SIZE, 1);
	Matrix* layer_2_biases = load_matrix_from_csv(LAYER_2_BIAS_FILE, LAYER_2_SIZE, 1);
	Matrix* layer_3_biases = load_matrix_from_csv(LAYER_3_BIAS_FILE, LAYER_3_SIZE, 1);

	MnistCSV mnist_data = {
		fopen("data/mnist/mnist_train.csv", "r"),
		NULL,
		NULL,
		0,
		0,
		NULL
	};
	mnist_csv_init(&mnist_data);

	for (int i = 0; i < num_epochs; i++) {
		// Set up this epoch
		double epoch_avg_accuracy = 0;
		double epoch_avg_loss = 0;
		float epoch_learn_rate = -SGD_LEARN_RATE_MULTIPLIER;
		int num_batches = ceil((float) mnist_data.num_examples / (float) SGD_BATCH_SIZE);

		// Reset the MnistCSV struct for a new round of SGD
		memset(mnist_data.sampled, 0, mnist_data.num_examples);
		mnist_data.num_sampled = 0;

		for (int j = 0; j < num_batches; j++) {
			int num_remaining = mnist_data.num_examples - j * SGD_BATCH_SIZE;
			int num_in_this_batch = num_remaining > SGD_BATCH_SIZE ? SGD_BATCH_SIZE : num_remaining;
			int num_correct = 0;
			double batch_loss = 0;
			matrix_float_t* input_data = malloc(num_in_this_batch * LAYER_INPUT_SIZE * sizeof(matrix_float_t));
			Matrix input_nodes = { LAYER_INPUT_SIZE, num_in_this_batch, input_data };
			matrix_float_t* expectations = malloc(num_in_this_batch * LAYER_3_SIZE * sizeof(matrix_float_t));
			Matrix output_nodes_expected = { LAYER_3_SIZE, num_in_this_batch, expectations };
			
			// Construct input and output matrices for this batch
			for (int k = 0; k < num_in_this_batch; k++) {
				MnistExample ex = get_random_data_take(&mnist_data);
				// visualize_digit_data(ex);

				for (int p = 0; p < LAYER_INPUT_SIZE; p++) {
					input_data[k + num_in_this_batch * p] = ex.X[p * ex.num_examples];
				}

				int expectation = (int) ex.y;
				for (int p = 0; p < LAYER_3_SIZE; p++) {
					expectations[k + num_in_this_batch * p] = 0;
				}
				expectations[k + expectation * num_in_this_batch] = 1;
			}
			matrix_scale(&input_nodes, 1 / 255.0F);

			// Forward pass
			Matrix* layer_1_activations_raw = matrix_multiply(*layer_1_weights, input_nodes);
			matrix_add_tile_columns(layer_1_activations_raw, layer_1_biases);
			Matrix* layer_1_activations = clone_matrix(*layer_1_activations_raw);
			relu(layer_1_activations->data, LAYER_1_SIZE * num_in_this_batch);
 
			Matrix* layer_2_activations_raw = matrix_multiply(*layer_2_weights, *layer_1_activations);
			matrix_add_tile_columns(layer_2_activations_raw, layer_2_biases);
			Matrix* layer_2_activations = clone_matrix(*layer_2_activations_raw);
			relu(layer_2_activations->data, LAYER_2_SIZE * num_in_this_batch);

			Matrix* layer_3_activations_raw = matrix_multiply(*layer_3_weights, *layer_2_activations);
			matrix_add_tile_columns(layer_3_activations_raw, layer_3_biases);
			Matrix* layer_3_activations = clone_matrix(*layer_3_activations_raw);
			softmax(layer_3_activations->data, LAYER_3_SIZE, num_in_this_batch);

			// Post forward pass
			for (int k = 0; k < num_in_this_batch; k++) {
				// Check if prediction is correct
				int prediction = 0;
				matrix_float_t max_confidence = 0;
				for (int p = 0; p < LAYER_3_SIZE; p++) {
					if (layer_3_activations->data[p * num_in_this_batch + k] > max_confidence) {
						max_confidence = layer_3_activations->data[p * num_in_this_batch + k];
						prediction = p;
					}
				}
				if (output_nodes_expected.data[k + prediction * output_nodes_expected.cols] == 1) {
					num_correct++;
				}

				// Compute loss for this batch
				matrix_float_t* expected_distribution = output_nodes_expected.data + k * LAYER_3_SIZE;
				matrix_float_t* actual_distribution = layer_3_activations->data + k * LAYER_3_SIZE;
				batch_loss += cross_entropy_loss(actual_distribution, expected_distribution, LAYER_3_SIZE);
			}
			epoch_avg_accuracy += (float) num_correct;
			epoch_avg_loss += batch_loss;

			// Backpropagation
			double scale = 1 / (double) LAYER_INPUT_SIZE;

			// Layer 3 parameters
			Matrix* layer_3_activation_raw_gradient = clone_matrix(*layer_3_activations);
			matrix_scale(&output_nodes_expected, -1.0F);
			matrix_add(layer_3_activation_raw_gradient, &output_nodes_expected);
			matrix_scale(&output_nodes_expected, -1.0F);
			matrix_transpose(layer_2_activations);
			matrix_scale(layer_3_activation_raw_gradient, scale);
			Matrix* layer_3_weight_gradient = matrix_multiply(*layer_3_activation_raw_gradient, *layer_2_activations);
			matrix_transpose(layer_2_activations);
			Matrix* layer_3_bias_gradient = matrix_col_sum(*layer_3_activation_raw_gradient);

			matrix_transpose(layer_3_weights);
			Matrix* layer_2_activation_gradient = matrix_multiply(*layer_3_weights, *layer_3_activation_raw_gradient);
			matrix_transpose(layer_3_weights);
			Matrix* layer_2_activation_raw_gradient = clone_matrix(*layer_2_activations_raw);
			relu_ddx(layer_2_activation_raw_gradient->data, layer_2_activation_raw_gradient->rows * layer_2_activation_raw_gradient->cols);
			matrix_multiply_elementwise(layer_2_activation_raw_gradient, layer_2_activation_gradient);
			matrix_transpose(layer_1_activations);
			Matrix* layer_2_weight_gradient = matrix_multiply(*layer_2_activation_raw_gradient, *layer_1_activations);
			matrix_transpose(layer_1_activations);
			Matrix* layer_2_bias_gradient = matrix_col_sum(*layer_2_activation_raw_gradient);

			matrix_transpose(layer_2_weights);
			Matrix* layer_1_activation_gradient = matrix_multiply(*layer_2_weights, *layer_2_activation_raw_gradient);
			matrix_transpose(layer_2_weights);
			Matrix* layer_1_activation_raw_gradient = clone_matrix(*layer_1_activations_raw);
			relu_ddx(layer_1_activation_raw_gradient->data, layer_1_activation_raw_gradient->rows * layer_1_activation_raw_gradient->cols);
			matrix_multiply_elementwise(layer_1_activation_raw_gradient, layer_1_activation_gradient);
			matrix_transpose(&input_nodes);
			Matrix* layer_1_weight_gradient = matrix_multiply(*layer_1_activation_raw_gradient, input_nodes);
			matrix_transpose(&input_nodes);
			Matrix* layer_1_bias_gradient = matrix_col_sum(*layer_1_activation_raw_gradient);

			// Update parameters
			clip_gradient(layer_3_weight_gradient);
			clip_gradient(layer_3_bias_gradient);
			clip_gradient(layer_2_weight_gradient);
			clip_gradient(layer_2_bias_gradient);
			clip_gradient(layer_1_weight_gradient);
			clip_gradient(layer_1_bias_gradient);

			matrix_scale(layer_3_weight_gradient, epoch_learn_rate);
			matrix_scale(layer_3_bias_gradient, epoch_learn_rate);
			matrix_scale(layer_2_weight_gradient, epoch_learn_rate);
			matrix_scale(layer_2_bias_gradient, epoch_learn_rate);
			matrix_scale(layer_1_weight_gradient, epoch_learn_rate);
			matrix_scale(layer_1_bias_gradient, epoch_learn_rate);

			matrix_add(layer_3_weights, layer_3_weight_gradient);
			matrix_add(layer_3_biases, layer_3_bias_gradient);
			matrix_add(layer_2_weights, layer_2_weight_gradient);
			matrix_add(layer_2_biases, layer_2_bias_gradient);
			matrix_add(layer_1_weights, layer_1_weight_gradient);
			matrix_add(layer_1_biases, layer_1_bias_gradient);

			// Free memory allocated in this batch
			free_matrix(layer_1_activations);
			free_matrix(layer_1_activations_raw);
			free_matrix(layer_2_activations);
			free_matrix(layer_2_activations_raw);
			free_matrix(layer_3_activations);
			free_matrix(layer_3_activations_raw);
			free_matrix(layer_1_activation_gradient);
			free_matrix(layer_1_activation_raw_gradient);
			free_matrix(layer_2_activation_gradient);
			free_matrix(layer_2_activation_raw_gradient);
			free_matrix(layer_3_activation_raw_gradient);
			free_matrix(layer_1_weight_gradient);
			free_matrix(layer_1_bias_gradient);
			free_matrix(layer_2_weight_gradient);
			free_matrix(layer_2_bias_gradient);
			free_matrix(layer_3_weight_gradient);
			free_matrix(layer_3_bias_gradient);
			free(input_data);
			free(expectations); 
		}

		epoch_avg_accuracy /= (float) mnist_data.num_examples;
		epoch_avg_loss /= (float) mnist_data.num_examples;
		printf("Epoch %d:\tAvg accuracy: %.3f\tAvg loss: %.5f\n", i, epoch_avg_accuracy, epoch_avg_loss);
	}

	// Convert matrix data back to float to save as CSV
	float* layer_1_weights_float_data = malloc(layer_1_weights->rows * layer_1_weights->cols * sizeof(float));
	float* layer_2_weights_float_data = malloc(layer_2_weights->rows * layer_2_weights->cols * sizeof(float));
	float* layer_3_weights_float_data = malloc(layer_3_weights->rows * layer_3_weights->cols * sizeof(float));
	float* layer_1_biases_float_data = malloc(layer_1_biases->rows * layer_1_biases->cols * sizeof(float));
	float* layer_2_biases_float_data = malloc(layer_2_biases->rows * layer_2_biases->cols * sizeof(float));
	float* layer_3_biases_float_data = malloc(layer_3_biases->rows * layer_3_biases->cols * sizeof(float));

	for (int i = 0; i < layer_1_weights->rows * layer_1_weights->cols; i++) {
		layer_1_weights_float_data[i] = (float) layer_1_weights->data[i];
	}
	for (int i = 0; i < layer_2_weights->rows * layer_2_weights->cols; i++) {
		layer_2_weights_float_data[i] = (float) layer_2_weights->data[i];
	}
	for (int i = 0; i < layer_3_weights->rows * layer_3_weights->cols; i++) {
		layer_3_weights_float_data[i] = (float) layer_3_weights->data[i];
	}
	for (int i = 0; i < layer_1_biases->rows * layer_1_biases->cols; i++) {
		layer_1_biases_float_data[i] = (float) layer_1_biases->data[i];
	}
	for (int i = 0; i < layer_2_biases->rows * layer_2_biases->cols; i++) {
		layer_2_biases_float_data[i] = (float) layer_2_biases->data[i];
	}
	for (int i = 0; i < layer_3_biases->rows * layer_3_biases->cols; i++) {
		layer_3_biases_float_data[i] = (float) layer_3_biases->data[i];
	}

	write_csv_contents(LAYER_1_WEIGHT_FILE, layer_1_weights_float_data, layer_1_weights->cols, layer_1_weights->rows);
	write_csv_contents(LAYER_2_WEIGHT_FILE, layer_2_weights_float_data, layer_2_weights->cols, layer_2_weights->rows);
	write_csv_contents(LAYER_3_WEIGHT_FILE, layer_3_weights_float_data, layer_3_weights->cols, layer_3_weights->rows);
	write_csv_contents(LAYER_1_BIAS_FILE, layer_1_biases_float_data, layer_1_biases->cols, layer_1_biases->rows);
	write_csv_contents(LAYER_2_BIAS_FILE, layer_2_biases_float_data, layer_2_biases->cols, layer_2_biases->rows);
	write_csv_contents(LAYER_3_BIAS_FILE, layer_3_biases_float_data, layer_3_biases->cols, layer_3_biases->rows);

	// Free memory needed during training
	free(layer_1_weights_float_data);
	free(layer_2_weights_float_data);
	free(layer_3_weights_float_data);
	free(layer_1_biases_float_data);
	free(layer_2_biases_float_data);
	free(layer_3_biases_float_data);
	free_matrix(layer_1_weights);
	free_matrix(layer_2_weights);
	free_matrix(layer_3_weights);
	free_matrix(layer_1_biases);
	free_matrix(layer_2_biases);
	free_matrix(layer_3_biases);
	free(mnist_data.X);
	free(mnist_data.y);
	free(mnist_data.sampled);
}

/**
 * Run the model (make `num_predictions` predictions)
 * 
 * If `num_predictions == -1`, run the model on the entire validation set
 */
void run(int num_predictions) {
	Matrix* layer_1_weights = load_matrix_from_csv(LAYER_1_WEIGHT_FILE, LAYER_1_SIZE, LAYER_INPUT_SIZE);
	Matrix* layer_2_weights = load_matrix_from_csv(LAYER_2_WEIGHT_FILE, LAYER_2_SIZE, LAYER_1_SIZE);
	Matrix* layer_3_weights = load_matrix_from_csv(LAYER_3_WEIGHT_FILE, LAYER_3_SIZE, LAYER_2_SIZE);
	Matrix* layer_1_biases = load_matrix_from_csv(LAYER_1_BIAS_FILE, LAYER_1_SIZE, 1);
	Matrix* layer_2_biases = load_matrix_from_csv(LAYER_2_BIAS_FILE, LAYER_2_SIZE, 1);
	Matrix* layer_3_biases = load_matrix_from_csv(LAYER_3_BIAS_FILE, LAYER_3_SIZE, 1);

	MnistCSV mnist_data = {
		fopen("data/mnist/mnist_test.csv", "r"),
		NULL,
		NULL,
		0,
		0,
		NULL
	};
	mnist_csv_init(&mnist_data);

	if (num_predictions == -1 || num_predictions > mnist_data.num_examples) {
		num_predictions = mnist_data.num_examples;
	}
	
	Matrix input_nodes = {LAYER_INPUT_SIZE, num_predictions, NULL};
	Matrix output_nodes = {LAYER_3_SIZE, num_predictions, NULL};
	input_nodes.data = malloc(LAYER_INPUT_SIZE * num_predictions * sizeof(matrix_float_t));
	output_nodes.data = malloc(LAYER_3_SIZE * num_predictions * sizeof(matrix_float_t));

	printf("Running predictions for %d digits...", num_predictions);
	fflush(stdout);

	int num_correct = 0;
	for (int i = 0; i < num_predictions; i++) {
		MnistExample ex = get_random_data_take(&mnist_data);
		// visualize_digit_data(ex);

		for (int j = 0; j < LAYER_INPUT_SIZE; j++) {
			input_nodes.data[i + j * num_predictions] = ex.X[j * ex.num_examples];
		}
		int expectation = (int) ex.y;
		for (int j = 0; j < LAYER_3_SIZE; j++) {
			output_nodes.data[i + j * num_predictions] = 0;
		}
		output_nodes.data[i + expectation * num_predictions] = 1;
	}
		
	// Pre-processing
	matrix_scale(&input_nodes, 1 / 255.0F);

	// Forward pass
	Matrix* layer_1_activations_raw = matrix_multiply(*layer_1_weights, input_nodes);
	matrix_add_tile_columns(layer_1_activations_raw, layer_1_biases);
	Matrix* layer_1_activations = clone_matrix(*layer_1_activations_raw);
	relu(layer_1_activations->data, LAYER_1_SIZE * num_predictions);
 
	Matrix* layer_2_activations_raw = matrix_multiply(*layer_2_weights, *layer_1_activations);
	matrix_add_tile_columns(layer_2_activations_raw, layer_2_biases);
	Matrix* layer_2_activations = clone_matrix(*layer_2_activations_raw);
	relu(layer_2_activations->data, LAYER_2_SIZE * num_predictions);

	Matrix* layer_3_activations_raw = matrix_multiply(*layer_3_weights, *layer_2_activations);
	matrix_add_tile_columns(layer_3_activations_raw, layer_3_biases);
	Matrix* layer_3_activations = clone_matrix(*layer_3_activations_raw);
	softmax(layer_3_activations->data, LAYER_3_SIZE, num_predictions);

	// Interpret results
	int prediction = 0;
	matrix_float_t max_confidence = 0;
	for (int i = 0; i < LAYER_3_SIZE; i++) {
		if (layer_3_activations->data[i] > max_confidence) {
			max_confidence = layer_3_activations->data[i];
			prediction = i;
		}
	}

	// Count number of correct predictions
	for (int i = 0; i < num_predictions; i++) {
		int prediction = 0;
		matrix_float_t max_confidence = 0;
		for (int j = 0; j < LAYER_3_SIZE; j++) {
			if (layer_3_activations->data[j * num_predictions + i] > max_confidence) {
				max_confidence = layer_3_activations->data[j * num_predictions + i];
				prediction = j;
			}
		}
		if (output_nodes.data[i + prediction * num_predictions] == 1) {
			num_correct++;
		}
	}

	printf("done! Got %d correct (%.3f).\n", num_correct, (float) num_correct / (float) num_predictions);
		
	// Free memory
	free_matrix(layer_1_activations);
	free_matrix(layer_2_activations);
	free_matrix(layer_3_activations);
	free_matrix(layer_1_activations_raw);
	free_matrix(layer_2_activations_raw);
	free_matrix(layer_3_activations_raw);
	free(input_nodes.data);
	free(output_nodes.data);
	free_matrix(layer_1_weights);
	free_matrix(layer_2_weights);
	free_matrix(layer_3_weights);
	free_matrix(layer_1_biases);
	free_matrix(layer_2_biases);
	free_matrix(layer_3_biases);
	free(mnist_data.X);
	free(mnist_data.y);
	free(mnist_data.sampled);
}

int main(int argc, char** argv) {
	srand(42);
	if (argc < 2) {
		printf("Please supply an argument, options:\n\trun [<num predictions>]\n\ttrain <num epochs>\n\tinit\n");
		exit(1);
	}
	if (strncmp(argv[1], "run", 3) == 0) {
		if (argc < 3) {	
			run(-1);
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
		printf("Unrecognized argument, options:\n\trun [<num predictions>]\n\ttrain <num epochs>\n\tinit\n");
		exit(1);
	}
}