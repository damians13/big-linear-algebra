#include "layer.h"
#include "matrix.h"
#include "csv.h"
#include <stdlib.h>

void feed_forward(struct Layer* l) {
	if (!l->has_previous_layer) {
		return;
	}
	struct Matrix* output = matrix_multiply(*l->weights, *l->previous_layer->nodes);
	matrix_add(output, l->biases);
	l->activation(output->data, l->num_nodes);
	if (l->has_nodes) {
		free(l->nodes);
	}
	l->nodes = output;
	l->has_nodes = 1;
}

void free_layer_data(struct Layer l)  {
	if (l.has_nodes) {
		free_matrix(l.nodes);
	}
	if (!l.has_previous_layer) {
		return;
	}
	free_matrix(l.weights);
	free_matrix(l.biases);
}

void load_weights_from_csv(struct Layer* l, const char* filepath) {
	if (!l->has_previous_layer) {
		return;
	}
	l->weights = make_matrix(l->num_nodes, l->previous_layer->num_nodes, read_csv_contents(filepath));
}

void load_biases_from_csv(struct Layer* l, const char* filepath) {
	if (!l->has_previous_layer) {
		return;
	}
	l->biases = make_matrix(l->num_nodes, 1, read_csv_contents(filepath));
}
