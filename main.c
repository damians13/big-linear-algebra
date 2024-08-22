#include "lib/matrix.h"
#include "lib/csv.h"
#include "lib/layer.h"
#include <stdlib.h>
#include <stdio.h>

void activation(float* data, int num) {
	for (int i = 0; i < num; i++) {
		data[i] *= 0.1;
	}
}

void activation_ddx(float* data, int num) {
	for (int i = 0; i < num; i++) {
		data[i] = 0.1;
	}
}

int main(int argc, char** argv) {
	float data[] = {
		1, 2, 3, 
		4, 5, 6
	};
	struct Matrix m1;
	m1.cols = 3;
	m1.rows = 2;
	m1.data = data;

	float _data[] = {
		1, 0.5,
		0.2, 1,
		0, 2
	};
	struct Matrix m2;
	m2.cols = 2;
	m2.rows = 3;
	m2.data = _data;

	struct Matrix* m3 = matrix_multiply(m1, m2);
	print_matrix(*m3);
	free_matrix(m3);

	float* v = read_csv_contents("data/a.csv");
	for (int i = 0; i < 9; i++) {
		printf("%.5f\n", v[i]);
	}
	free(v);

	float d[] = {1, 2.3, 4.567, 0, 0, 0};
	write_csv_contents("data/b.csv", d, 3, 2);

	struct Layer input;
	input.num_nodes = 3;
	input.has_previous_layer = 0;
	input.nodes = make_matrix(3, 1, read_csv_contents("data/inputs.csv"));
	input.has_nodes = 1;

	struct Layer hidden = { 2, 0, 0, 0, 0, &input, &activation, &activation_ddx, 1, 0 };
	load_weights_from_csv(&hidden, "data/weights.csv");
	load_biases_from_csv(&hidden, "data/biases.csv");

	struct Layer output;
	output.num_nodes = 2;
	output.previous_layer = &hidden;
	output.has_previous_layer = 1;
	output.has_nodes = 0;
	output.activation = &activation;
	output.activation_ddx = &activation_ddx;
	load_weights_from_csv(&output, "data/weights.csv");
	load_biases_from_csv(&output, "data/biases.csv");

	feed_forward(&hidden);
	feed_forward(&output);
	print_matrix(*output.nodes);

	print_matrix(*output.weights);
	print_matrix(*output.biases);

	float expectations[] = { 0.5, 0.5 };
	back_propagate_errors(&output, expectations, 0.05);

	print_matrix(*output.weights);
	print_matrix(*output.biases);

	free_layer_data(output);
	free_layer_data(hidden);
	free_matrix(input.nodes);
}
