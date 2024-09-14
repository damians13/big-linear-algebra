#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/layer.h"
#include "../lib/mnist_csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

void run(int num) {
	struct Matrix input_nodes = { 784, 1, 0 };
	struct Matrix hidden_weights = { 20, 784, read_csv_contents("data/mnist/hidden_weights.csv") };
	struct Matrix hidden_biases = { 20, 1, read_csv_contents("data/mnist/hidden_biases.csv") };
	struct Matrix output_weights = { 10, 20, read_csv_contents("data/mnist/output_weights.csv") };
	struct Matrix output_biases = { 10, 1, read_csv_contents("data/mnist/output_biases.csv") };
	
	struct Layer input = { 784, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 1 };
	struct Layer hidden = { 20, 0, 0, &hidden_weights, &hidden_biases, &input, softmax, softmax_ddx, 1, 0 };
	struct Layer output = { 10, 0, 0, &output_weights, &output_biases, &hidden, softmax, softmax_ddx, 1, 0 };

	struct MnistCSV testing_data = {
		fopen("data/mnist/mnist_test.csv", "r"),
		malloc(785 * sizeof(int))
	};
	int num_correct = 0;

	for (int i = 0; i < num; i++) {
		get_next_data(&testing_data);
		visualize_digit_data(&testing_data);
		float expectation[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		expectation[testing_data.buffer[0]] = 1;
		input_nodes.data = (float*) testing_data.buffer + 1;

		feed_forward(&hidden);
		feed_forward(&output);
		
		int prediction_index = 0;
		float max_prediction = 0;
		printf("Predictions:\n");
		for (int i = 0; i < 10; i++) {
			if (output.nodes->data[i] > max_prediction) {
				max_prediction = output.nodes->data[i];
				prediction_index = i;
			}
			printf("\t%d: %.2f\n", i, output.nodes->data[i]);
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
			printf("Correct");
			num_correct++;
		} else {
			printf("Incorrect");
		}
		printf(" with cost: %.2f\n", cost);
	}

	float success_percent = (float) num_correct / (float) num;
	printf("\nGot %d correct out of %d, (%.2f%%)\n", num_correct, num, success_percent);

	free(testing_data.buffer);
	fclose(testing_data.file);
}

void train(int iterations, float learn_rate) {
	struct Matrix input_nodes = { 784, 1, 0 };
	struct Matrix hidden_weights = { 20, 784, read_csv_contents("data/mnist/hidden_weights.csv") };
	struct Matrix hidden_biases = { 20, 1, read_csv_contents("data/mnist/hidden_biases.csv") };
	struct Matrix output_weights = { 10, 20, read_csv_contents("data/mnist/output_weights.csv") };
	struct Matrix output_biases = { 10, 1, read_csv_contents("data/mnist/output_biases.csv") };
	
	struct Layer input = { 784, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 1 };
	struct Layer hidden = { 20, 0, 0, &hidden_weights, &hidden_biases, &input, softmax, softmax_ddx, 1, 0 };
	struct Layer output = { 10, 0, 0, &output_weights, &output_biases, &hidden, softmax, softmax_ddx, 1, 0 };

	struct MnistCSV training_data = {
		fopen("data/mnist/mnist_train.csv", "r"),
		malloc(785 * sizeof(int))
	};
	int report_costs_every_n = 5;
	float prev_costs[report_costs_every_n];
	for (int i = 0; i < iterations; i++) {
		get_next_data(&training_data);
		float expectation[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		expectation[training_data.buffer[0]] = 1;
		input_nodes.data = (float*) training_data.buffer + 1;

		feed_forward(&hidden);
		feed_forward(&output);

		prev_costs[i % report_costs_every_n] = (expectation[0] - output.nodes->data[0]) * (expectation[0] - output.nodes->data[0])
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
		
		if (i % report_costs_every_n == report_costs_every_n - 1) {
			float avg = 0;
			printf("Last %d costs:\n", report_costs_every_n);
			for (int j = 0; j < report_costs_every_n; j++) {
				avg += prev_costs[j];
				printf("\tCost[%d]: %.3f\n", j, prev_costs[j]);
			}
			avg /= report_costs_every_n;
			printf("\tAvg: %.3f\n", avg);
		}
	}

	write_csv_contents("data/mnist/hidden_weights.csv", hidden.weights->data, 784, 20);
	write_csv_contents("data/mnist/hidden_biases.csv", hidden.biases->data, 1, 20);
	write_csv_contents("data/mnist/output_weights.csv", output.weights->data, 20, 10);
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
	float data[20 * 784];
	for (int i = 0; i < 20 * 784; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_weights.csv", data, 784, 20);

	for (int i = 0; i < 20; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/hidden_biases.csv", data, 1, 20);

	for (int i = 0; i < 10 * 20; i++) {
		data[i] = (float)rand()/(float)(RAND_MAX) - 0.5;
	}
	write_csv_contents("data/mnist/output_weights.csv", data, 20, 10);

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
			printf("Please supply a number of samples to use, usage:\n\run <num>\n");
			exit(1);
		}
		run(atoi(argv[2]));
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