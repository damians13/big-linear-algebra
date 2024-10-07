#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/layer.h"
#include "../lib/mnist_csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define HIDDEN_LAYER_SIZE 200
#define TRAINING_REPORT_COSTS_EVERY_N 20

void relu(float* data, int num) {
	for (int i = 0; i < num; i++) {
		if (data[i] < 0) {
			data[i] = 0;
		}
	}
}

void relu_ddx(float* data, int num) {
	for (int i = 0; i < num; i++) {
		data[i] = data[i] > 0 ? 1 : 0;
	}
}

void softmax(float* data, int num) {
	float sum_of_exponents = 0;
	for (int i = 0; i < num; i++) {
		sum_of_exponents += exp(data[i]);
	}
	for (int i = 0; i < num; i++) {
		data[i] /= sum_of_exponents;
	}
}

void softmax_ddx(float* data, int num) {
	float sum_of_exponents = 0;
	for (int i = 0; i < num; i++) {
		sum_of_exponents += exp(data[i]);
	}
	for (int i = 0; i < num; i++) {
		float e_to_the_x_i = exp(data[i]);
		data[i] = (sum_of_exponents * e_to_the_x_i - e_to_the_x_i * e_to_the_x_i) / (sum_of_exponents * sum_of_exponents);
	}
}

void run(int num, int report_every_n) {
	struct Matrix input_nodes = { 784, 1, 0 };
	struct Matrix hidden_weights = { HIDDEN_LAYER_SIZE, 784, read_csv_contents("data/mnist/hidden_weights.csv") };
	struct Matrix hidden_biases = { HIDDEN_LAYER_SIZE, 1, read_csv_contents("data/mnist/hidden_biases.csv") };
	struct Matrix hidden_weights_2 = { HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, read_csv_contents("data/mnist/hidden_weights_2.csv") };
	struct Matrix hidden_biases_2 = { HIDDEN_LAYER_SIZE, 1, read_csv_contents("data/mnist/hidden_biases_2.csv") };
	struct Matrix output_weights = { 10, HIDDEN_LAYER_SIZE, read_csv_contents("data/mnist/output_weights.csv") };
	struct Matrix output_biases = { 10, 1, read_csv_contents("data/mnist/output_biases.csv") };
	
	struct Layer input = { 784, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 1 };
	struct Layer hidden = { HIDDEN_LAYER_SIZE, 0, 0, &hidden_weights, &hidden_biases, &input, relu, relu_ddx, 1, 0 };
	struct Layer hidden2 = { HIDDEN_LAYER_SIZE, 0, 0, &hidden_weights_2, &hidden_biases_2, &hidden, relu, relu_ddx, 1, 0 };
	struct Layer output = { 10, 0, 0, &output_weights, &output_biases, &hidden2, softmax, softmax_ddx, 1, 0 };

	struct MnistCSV testing_data = {
		fopen("data/mnist/mnist_test.csv", "r"),
		malloc(785 * sizeof(int))
	};
	int num_correct = 0;

	for (int i = 0; i < num; i++) {
		char report = i % report_every_n == report_every_n - 1;

		get_next_data(&testing_data);
		if (report) {
			visualize_digit_data(&testing_data);
		}
		float expectation[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		expectation[testing_data.buffer[0]] = 1;
		input_nodes.data = (float*) testing_data.buffer + 1;
		matrix_scale(&input_nodes, 1 / 255.0F);

		feed_forward(&hidden);
		feed_forward(&hidden2);
		feed_forward(&output);
		
		int prediction_index = 0;
		float max_prediction = 0;
		if (report) {
			printf("Predictions:\n");
		}
		for (int i = 0; i < 10; i++) {
			if (output.nodes->data[i] > max_prediction) {
				max_prediction = output.nodes->data[i];
				prediction_index = i;
			}
			if (report) {
				printf("\t%d: %.2f\n", i, output.nodes->data[i]);
			}
		}

		float cost = (expectation[0] - output.nodes->data[0]) * (expectation[0] - output.nodes->data[0])
				   + (expectation[1] - output.nodes->data[1]) * (expectation[1] - output.nodes->data[1])
				   + (expectation[2] - output.nodes->data[2]) * (expectation[2] - output.nodes->data[2])
				   + (expectation[3] - output.nodes->data[3]) * (expectation[3] - output.nodes->data[3])
				   + (expectation[4] - output.nodes->data[4]) * (expectation[4] - output.nodes->data[4])
				   + (expectation[5] - output.nodes->data[5]) * (expectation[5] - output.nodes->data[5])
				   + (expectation[6] - output.nodes->data[6]) * (expectation[6] - output.nodes->data[6])
				   + (expectation[7] - output.nodes->data[7]) * (expectation[7] - output.nodes->data[7])
				   + (expectation[8] - output.nodes->data[8]) * (expectation[8] - output.nodes->data[8])
				   + (expectation[9] - output.nodes->data[9]) * (expectation[9] - output.nodes->data[9]);

		if (prediction_index + 1 == testing_data.buffer[0]) {
			if (report) {
				printf("Correct");
			}
			num_correct++;
		} else {
			if (report) {
				printf("Incorrect");
			}
		}
		if (report) {
			printf(" with cost: %.2f\n", cost);
		}
	}

	float success_percent = (float) num_correct / (float) num;
	printf("\nGot %d correct out of %d, (%.2f%%)\n", num_correct, num, success_percent);

	free(testing_data.buffer);
	fclose(testing_data.file);
}

void train(int iterations, float learn_rate, int shouldOutput) {
	struct Matrix input_nodes = { 784, 1, 0 };
	struct Matrix hidden_weights = { HIDDEN_LAYER_SIZE, 784, read_csv_contents("data/mnist/hidden_weights.csv") };
	struct Matrix hidden_biases = { HIDDEN_LAYER_SIZE, 1, read_csv_contents("data/mnist/hidden_biases.csv") };
	struct Matrix hidden_weights_2 = { HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, read_csv_contents("data/mnist/hidden_weights_2.csv") };
	struct Matrix hidden_biases_2 = { HIDDEN_LAYER_SIZE, 1, read_csv_contents("data/mnist/hidden_biases_2.csv") };
	struct Matrix output_weights = { 10, HIDDEN_LAYER_SIZE, read_csv_contents("data/mnist/output_weights.csv") };
	struct Matrix output_biases = { 10, 1, read_csv_contents("data/mnist/output_biases.csv") };
	
	struct Layer input = { 784, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 1 };
	struct Layer hidden = { HIDDEN_LAYER_SIZE, 0, 0, &hidden_weights, &hidden_biases, &input, relu, relu_ddx, 1, 0 };
	struct Layer hidden2 = { HIDDEN_LAYER_SIZE, 0, 0, &hidden_weights_2, &hidden_biases_2, &hidden, relu, relu_ddx, 1, 0 };
	struct Layer output = { 10, 0, 0, &output_weights, &output_biases, &hidden2, softmax, softmax_ddx, 1, 0 };

	struct MnistCSV training_data = {
		fopen("data/mnist/mnist_train.csv", "r"),
		malloc(785 * sizeof(int))
	};
	float prev_costs[TRAINING_REPORT_COSTS_EVERY_N];
	for (int i = 0; i < iterations; i++) {
		get_next_data(&training_data);
		float expectation[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		expectation[training_data.buffer[0]] = 1;
		input_nodes.data = (float*) training_data.buffer + 1;
		matrix_scale(&input_nodes, 1 / 255.0F);

		feed_forward(&hidden);
		feed_forward(&hidden2);
		feed_forward(&output);

		prev_costs[i % TRAINING_REPORT_COSTS_EVERY_N] = (expectation[0] - output.nodes->data[0]) * (expectation[0] - output.nodes->data[0])
							   				 + (expectation[1] - output.nodes->data[1]) * (expectation[1] - output.nodes->data[1])
							   				 + (expectation[2] - output.nodes->data[2]) * (expectation[2] - output.nodes->data[2])
							   				 + (expectation[3] - output.nodes->data[3]) * (expectation[3] - output.nodes->data[3])
							   				 + (expectation[4] - output.nodes->data[4]) * (expectation[4] - output.nodes->data[4])
							   				 + (expectation[5] - output.nodes->data[5]) * (expectation[5] - output.nodes->data[5])
							   				 + (expectation[6] - output.nodes->data[6]) * (expectation[6] - output.nodes->data[6])
							   				 + (expectation[7] - output.nodes->data[7]) * (expectation[7] - output.nodes->data[7])
							   				 + (expectation[8] - output.nodes->data[8]) * (expectation[8] - output.nodes->data[8])
							   				 + (expectation[9] - output.nodes->data[9]) * (expectation[9] - output.nodes->data[9]);

		back_propagate_errors(&output, expectation, learn_rate);
		
		if (shouldOutput && i % TRAINING_REPORT_COSTS_EVERY_N == TRAINING_REPORT_COSTS_EVERY_N - 1) {
			float avg = 0;
			printf("Last %d costs:\n", TRAINING_REPORT_COSTS_EVERY_N);
			for (int j = 0; j < TRAINING_REPORT_COSTS_EVERY_N; j++) {
				avg += prev_costs[j];
				printf("\tCost[%d]: %.3f\n", j, prev_costs[j]);
			}
			avg /= TRAINING_REPORT_COSTS_EVERY_N;
			printf("\tAvg: %.3f\n", avg);
		}
		if (i == iterations - 1 && !shouldOutput) {
			float avg = 0;
			for (int j = 0; j < TRAINING_REPORT_COSTS_EVERY_N; j++) {
				avg += prev_costs[j];
			}
			avg /= TRAINING_REPORT_COSTS_EVERY_N;
			printf("Final batch avg: %.3f\n", avg);
		}
	}

	write_csv_contents("data/mnist/hidden_weights_2.csv", hidden2.weights->data, 784, HIDDEN_LAYER_SIZE);
	write_csv_contents("data/mnist/hidden_biases_2.csv", hidden2.biases->data, 1, HIDDEN_LAYER_SIZE);
	write_csv_contents("data/mnist/hidden_weights.csv", hidden.weights->data, 784, HIDDEN_LAYER_SIZE);
	write_csv_contents("data/mnist/hidden_biases.csv", hidden.biases->data, 1, HIDDEN_LAYER_SIZE);
	write_csv_contents("data/mnist/output_weights.csv", output.weights->data, HIDDEN_LAYER_SIZE, 10);
	write_csv_contents("data/mnist/output_biases.csv", output.biases->data, 1, 10);

	printf("Finished training\n");

	free_matrix(output.nodes);
	free_matrix(output.raw_nodes);
	free_matrix(hidden.nodes);
	free_matrix(hidden.raw_nodes);

	free_matrix_data(&hidden_weights);
	free_matrix_data(&hidden_biases);
	free_matrix_data(&output_weights);
	free_matrix_data(&output_biases);

	free(training_data.buffer);
	fclose(training_data.file);
}

void init() {
	float data[HIDDEN_LAYER_SIZE * 784];
	for (int i = 0; i < HIDDEN_LAYER_SIZE * 784; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_weights.csv", data, 784, HIDDEN_LAYER_SIZE);

	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_biases.csv", data, 1, HIDDEN_LAYER_SIZE);
	
	for (int i = 0; i < HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_weights_2.csv", data, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);

	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_biases_2.csv", data, 1, HIDDEN_LAYER_SIZE);

	for (int i = 0; i < 10 * HIDDEN_LAYER_SIZE; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/output_weights.csv", data, HIDDEN_LAYER_SIZE, 10);

	for (int i = 0; i < 10; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/output_biases.csv", data, 1, 10);
}

/**
 * This model will classify hand-drawn digits from the MNIST dataset
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
			run(atoi(argv[2]), 1);
		} else {
			run(atoi(argv[2]), atoi(argv[3]));
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