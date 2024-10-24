// MNIST hinge loss experiment
#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/layer.h"
#include "../lib/mnist_csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define FILEPATH_X_OFFSET 25

void init() {
	float data[784];
	char filepath[] = "data/mnist_hinge/weights_x.csv";
	srand(42);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 784; j++) {
			data[j] = (float)rand()/(10 * (float)(RAND_MAX)) - 0.05;
		}
		filepath[FILEPATH_X_OFFSET] = i + '0'; // Replace 'x' with i in the above string
		write_csv_contents(filepath, data, 1, 784);
	}
}

float norm(struct Matrix m) {
	float total = 0.0f;
	for (int i = 0; i < m.rows * m.cols; i++) {
		total += m.data[i] * m.data[i];
	}
	return sqrt(total);
}

void train(int iterations, float learn_rate) {
	char filepath[] = "data/mnist_hinge/weights_x.csv";
	struct Matrix input_nodes = { 784, 1, 0};
	struct Matrix ensemble_weights[10];
	for (int i = 0; i < 10; i++) {
		filepath[FILEPATH_X_OFFSET] = i + '0';
		struct Matrix m = { 1, 784, read_csv_contents(filepath) }; // Stored as 1 x 784 for multiplication later
		ensemble_weights[i] = m;
	}

	struct MnistCSV training_data = {
		fopen("data/mnist/mnist_train.csv", "r"),
		malloc(785 * sizeof(float))
	};
	int num_correct = 0;
	// int num_training_examples = count_num_lines(training_data.file);
	int num_training_examples = 5000; //TODO undo
	rewind(training_data.file);

	struct Matrix gradients[10];
	for (int i = 0; i < 10; i++) {
		gradients[i] = (struct Matrix) { 784, 1, malloc(784 * sizeof(float)) };
	}

	for (int i = 0; i < iterations; i++) {
		// Reset the gradients
		for (int j = 0; j < 10; j++) {
			memset(gradients[j].data, 0, 784);
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
				struct Matrix* output = matrix_multiply(ensemble_weights[p], input_nodes);
				float val = 1 - expectation[p] * output->data[0];
				free_matrix(output);
				for (int k = 0; k < 784; k++) {
					// Hinge loss: sum from j=1 to num_examples of max(0, 1 - y_j * w^T * x_j)
					// Derivative w.r.t weight k: sum from j=1 to num_examples of { 0 if 1 <= y_j * w^T * x_j OR -y_j * x_jk otherwise }
					if (val < 1) {
						gradients[p].data[k] += -expectation[p] * input_nodes.data[k];
					}
				}
			}
		}

		// printf("Gradient norms after iteration %d:\n", i);
		// Update weights for each model
		for (int j = 0; j < 10; j++) {
			// printf("\tModel %d: %.5f\n", j, norm(gradients[j]));
			matrix_scale(&gradients[j], learn_rate);
			matrix_add(&ensemble_weights[j], &gradients[j]);
		}
		
		// End of the iteration, go back to the start of the training data file
		rewind(training_data.file);
	}

	for (int i = 0; i < 10; i++) {
		filepath[FILEPATH_X_OFFSET] = i + '0';
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
			printf("Please supply a number of iterations and a learn rate, usage:\n\ttrain <iterations> <learn_rate>\n");
			exit(1);
		}
		train(atoi(argv[2]), atof(argv[3]));
	}else if (strncmp(argv[1], "init", 4) == 0) {
		init();
	} else {
		printf("Unrecognized argument, options:\n\trun\n\ttrain <iterations> <learn_rate>\n\tinit\n");
		exit(1);
	}
}