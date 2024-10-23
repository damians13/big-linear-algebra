// MNIST hinge loss experiment
#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/layer.h"
#include "../lib/mnist_csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void init() {
	float data[784];
	char filepath[] = "data/mnist_hinge/weights_x.csv";
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 784; j++) {
			data[j] = (float)rand()/(float)(RAND_MAX) - 0.5;
		}
		filepath[23] = i + '0'; // Replace 'x' with i in the above string
		write_csv_contents(filepath, data, 1, 784);
	}
}

void train(int iterations, float learn_rate, int num) {
	char filepath[] = "data/mnist_hinge/weights_x.csv";
	struct Matrix input_nodes = { 784, 1, 0};
	struct Matrix ensemble_weights[10];
	for (int i = 0; i < 10; i++) {
		filepath[23] = i + '0';
		struct Matrix m = { 1, 784, read_csv_contents(filepath) }; // Stored as 1 x 784 for multiplication later
		ensemble_weights[i] = m;
	}

	struct MnistCSV training_data = {
		fopen("data/mnist/mnist_train.csv", "r"),
		malloc(785 * num * sizeof(float))
	};
	int num_correct = 0;
	int num_training_examples = count_num_lines(training_data.file);
	rewind(training_data.file);

	struct Matrix gradients[10];
	for (int i = 0; i < 10; i++) {
		gradients[i] = (struct Matrix) { 784, 1, malloc(784 * sizeof(float)) };
	}

	for (int i = 0; i < iterations; i++) {
		// Reset the gradients
		for (int j = 0; j < 10; j++) {
			for (int k = 0; k < 784; k++) {
				gradients[j].data[k] = 0;
			}
		}

		for (int j = 0; j < num_training_examples; j++) {
			// Prep for this iteration
			get_next_data(&training_data);
			float expectation[] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
			expectation[(int) training_data.buffer[0]] = 1;
			input_nodes.data = training_data.buffer + 1;
			matrix_scale(&input_nodes, 1 / 255.0F);

			for (int p = 0; p < 10; p++) {
				// Compute this example's contribution to the gradient (for each weight)
				for (int k = 0; k < 784; k++) {
					// Hinge loss: max(0, 1 - y_i * o_i) for o_i = w^T * x_i
					struct Matrix* output = matrix_multiply(ensemble_weights[p], input_nodes);
					float val = 1 - expectation[p] * output->data[0];
					val = val < 0 ? 0 : val;
					// Add this example's impact to gradient update tally
					gradients[p].data[k] += val;

					free_matrix(output);
				}
			}
		}

		// Update weights for each model
		for (int i = 0; i < 10; i++) {
			matrix_scale(&gradients[i], -1 * learn_rate);
			matrix_add(&ensemble_weights[i], &gradients[i]);
		}
	}

	for (int i = 0; i < 10; i++) {
		filepath[23] = i + '0';
		write_csv_contents(filepath, ensemble_weights[i].data, 1, 784);
		free_matrix_data(&ensemble_weights[i]);
		free_matrix_data(&gradients[i]);
	}

	printf("Finished training\n");

	free(training_data.buffer);
	fclose(training_data.file);
}

/**
 * An ensemble model made up of 10 logistic regression models (1 for each digit) to serve as binary classifiers for MNIST data
 */
int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Please supply an argument, options:\n\trun\n\ttrain <iterations> <learn_rate>\n\tinit\n");
		exit(1);
	}
	if (strncmp(argv[1], "run", 3) == 0) {
		if (argc < 3) {
			printf("Please supply a number of samples to use, usage:\n\run <num> [<output_every_n = 1>]\n");
			exit(1);
		}
		if (argc < 4) {
			// run(atoi(argv[2]), 1);
		} else {
			// run(atoi(argv[2]), atoi(argv[3]));
		}
	} else if (strncmp(argv[1], "train", 5) == 0) {
		if (argc < 4) {
			printf("Please supply a number of iterations and a learn rate, usage:\n\ttrain <iterations> <learn_rate> [<output=1>]\n");
			exit(1);
		}
		if (argc < 5) {
			train(atoi(argv[2]), atof(argv[3]), 1);
		} else {
			train(atoi(argv[2]), atof(argv[3]), atoi(argv[4]));
		}
	}else if (strncmp(argv[1], "init", 4) == 0) {
		init();
	} else {
		printf("Unrecognized argument, options:\n\trun\n\ttrain <iterations> <learn_rate>\n\tinit\n");
		exit(1);
	}
}