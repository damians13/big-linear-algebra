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

	float* v = read_csv_contents("store/a.csv");
	for (int i = 0; i < 9; i++) {
		printf("%.5f\n", v[i]);
	}
	free(v);

	float d[] = {1, 2.3, 4.567, 0, 0, 0};
	write_csv_contents("store/b.csv", d, 3, 2);

	struct Layer input;
	input.num_nodes = 3;
	input.has_previous_layer = 0;
	input.nodes = make_matrix(3, 1, read_csv_contents("store/inputs.csv"));
	input.has_nodes = 1;

	struct Layer output;
	output.num_nodes = 2;
	output.previous_layer = &input;
	output.has_previous_layer = 1;
	output.has_nodes = 0;
	output.activation = &activation;
	load_weights_from_csv(&output, "store/weights.csv");
	load_biases_from_csv(&output, "store/biases.csv");

	feed_forward(&output);
	print_matrix(*output.nodes);

	free_layer_data(output);
	free_layer_data(input);
}
