#include "../lib/matrix.h"
#include "../lib/csv.h"
#include "../lib/layer.h"
#include <stdlib.h>
#include <stdio.h>

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

void run() {
	struct Matrix input_nodes = { 2, 1, read_csv_contents("../data/my_first_model/input_nodes.csv") };
	struct Matrix hidden_weights = { 3, 2, read_csv_contents("../data/my_first_model/hidden_weights.csv") };
	struct Matrix hidden_biases = { 3, 1, read_csv_contents("../data/my_first_model/hidden_biases.csv") };
	struct Matrix output_weights = { 2, 3, read_csv_contents("../data/my_first_model/output_weights.csv") };
	struct Matrix output_biases = { 2, 1, read_csv_contents("../data/my_first_model/output_biases.csv") };
	
	struct Layer input = { 2, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 0 };
	struct Layer hidden = { 3, 0, 0, &hidden_weights, &hidden_biases, &input, relu, relu_ddx, 1, 0 };
	struct Layer output = { 2, 0, 0, &output_weights, &output_biases, &hidden, relu, relu_ddx, 1, 0 };

	feed_forward(&hidden);
	feed_forward(&output);

	print_matrix(*output.nodes);

	if (output.nodes->data[0] > output.nodes->data[1]) {
		printf("Same sign!\n");
	} else {
		printf("Different signs!\n");
	}

	free_matrix(output.nodes);
	free_matrix(output.raw_nodes);
	free_matrix(hidden.nodes);
	free_matrix(hidden.raw_nodes);

	free_matrix_data(&input_nodes);
	free_matrix_data(&hidden_weights);
	free_matrix_data(&hidden_biases);
	free_matrix_data(&output_weights);
	free_matrix_data(&output_biases);
}

void train(int iterations, float learn_rate) {
	float* input_nodes_data = malloc(2 * sizeof(float));
	struct Matrix input_nodes = { 2, 1, input_nodes_data };
	struct Matrix hidden_weights = { 3, 2, read_csv_contents("../data/my_first_model/hidden_weights.csv") };
	struct Matrix hidden_biases = { 3, 1, read_csv_contents("../data/my_first_model/hidden_biases.csv") };
	struct Matrix output_weights = { 2, 3, read_csv_contents("../data/my_first_model/output_weights.csv") };
	struct Matrix output_biases = { 2, 1, read_csv_contents("../data/my_first_model/output_biases.csv") };
	
	struct Layer input = { 2, &input_nodes, 0, 0, 0, 0, 0, 0, 0, 0 };
	struct Layer hidden = { 3, 0, 0, &hidden_weights, &hidden_biases, &input, relu, relu_ddx, 1, 0 };
	struct Layer output = { 2, 0, 0, &output_weights, &output_biases, &hidden, relu, relu_ddx, 1, 0 };

	char filename[50];
	int report_costs_every_n = 10;
	float prev_costs[report_costs_every_n];
	for (int i = 0; i < iterations; i++) {
		switch (i % 4) {
			case 0:
				input_nodes.data[0] = (float)rand()/(float)(RAND_MAX);
				input_nodes.data[1] = (float)rand()/(float)(RAND_MAX);
				break;
			case 1:
				input_nodes.data[0] = -(float)rand()/(float)(RAND_MAX);
				input_nodes.data[1] = (float)rand()/(float)(RAND_MAX);
				break;
			case 2:
				input_nodes.data[0] = (float)rand()/(float)(RAND_MAX);
				input_nodes.data[1] = -(float)rand()/(float)(RAND_MAX);
				break;
			case 3:
				input_nodes.data[0] = -(float)rand()/(float)(RAND_MAX);
				input_nodes.data[1] = -(float)rand()/(float)(RAND_MAX);
				break;
		}

		float expectation[2];
		if (i % 2 == 0) {
			expectation[0] = 1;
			expectation[1] = 0;
		} else {
			expectation[0] = 0;
			expectation[1] = 1;
		}

		feed_forward(&hidden);
		feed_forward(&output);

		prev_costs[i % report_costs_every_n] = (expectation[0] - output.nodes->data[0]) * (expectation[0] - output.nodes->data[0])
							   				 + (expectation[1] - output.nodes->data[1]) * (expectation[1] - output.nodes->data[1]);

		back_propagate_errors(&output, expectation, 0.05);
		
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

		free(input_nodes.data);
	}

	float dummyData[] = { 0, 0 };
	write_csv_contents("../data/my_first_model/input_nodes.csv", dummyData, 1, 2);
	write_csv_contents("../data/my_first_model/hidden_weights.csv", hidden.weights->data, 2, 3);
	write_csv_contents("../data/my_first_model/hidden_biases.csv", hidden.biases->data, 1, 3);
	write_csv_contents("../data/my_first_model/output_weights.csv", output.weights->data, 3, 2);
	write_csv_contents("../data/my_first_model/output_biases.csv", output.biases->data, 1, 2);

	printf("Finished training");

	free_matrix(output.nodes);
	free_matrix(output.raw_nodes);
	free_matrix(hidden.nodes);
	free_matrix(hidden.raw_nodes);

	free_matrix_data(&input_nodes);
	free_matrix_data(&hidden_weights);
	free_matrix_data(&hidden_biases);
	free_matrix_data(&output_weights);
	free_matrix_data(&output_biases);

	free(input_nodes_data);
}

/**
 * This model will take in two numbers. The output should be *close to* [1, 0]
 * if they are both the same sign, or [0, 1] if they are different signs.
 */
int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Please supply an argument, options:\n\trun\n\ttrain <iterations> <learn_rate>\n");
		exit(1);
	}
	if (argv[1] == "run") {
		run();
	} else if (argv[1] == "train") {
		if (argc < 4) {
			printf("Please supply a number of iterations and a learn rate, usage:\n\ttrain <iterations> <learn_rate>\n");
			exit(1);
		}
		train(atoi(argv[2]), atof(argv[3]));
	} else {
		printf("Unrecognized argument, options:\n\trun\n\ttrain <iterations> <learn_rate>\n");
		exit(1);
	}
}