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
	if (l->has_nodes) {
		free(l->raw_nodes);
		free(l->nodes);
	}
	l->raw_nodes = clone_matrix(*output);
	l->activation(output->data, l->num_nodes);
	l->nodes = output;
	l->has_nodes = 1;
}

void free_layer_data(struct Layer l)  {
	if (l.has_nodes) {
		free_matrix(l.raw_nodes);
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

void do_back_propagate_errors(struct Layer* l, struct Layer* next_layer, struct Matrix* cost_ddx_next_layer_activation, float learn_rate) {
	if (!l->has_previous_layer) {
		return;
	}

	struct Matrix* activation_ddx_of_next_layer_raw_nodes = clone_matrix(*next_layer->raw_nodes);
	next_layer->activation_ddx(activation_ddx_of_next_layer_raw_nodes->data, next_layer->num_nodes);
	matrix_multiply_elementwise(activation_ddx_of_next_layer_raw_nodes, cost_ddx_next_layer_activation);
	matrix_transpose(next_layer->weights);
	struct Matrix* cost_ddx_current_layer_activation = matrix_multiply(*next_layer->weights, *activation_ddx_of_next_layer_raw_nodes);
	matrix_transpose(next_layer->weights);
	free_matrix(activation_ddx_of_next_layer_raw_nodes);

	struct Matrix* biases_change = clone_matrix(*l->raw_nodes);
	l->activation_ddx(biases_change->data, l->num_nodes);
	matrix_multiply_elementwise(biases_change, cost_ddx_current_layer_activation);
	matrix_scale(biases_change, -learn_rate);

	matrix_transpose(l->previous_layer->nodes);
	struct Matrix* weights_change = matrix_multiply(*biases_change, *l->previous_layer->nodes);
	matrix_transpose(l->previous_layer->nodes);

	do_back_propagate_errors(l->previous_layer, l, cost_ddx_current_layer_activation, learn_rate);

	matrix_add(l->weights, weights_change);
	matrix_add(l->biases, biases_change);

	free_matrix(weights_change);
	free_matrix(biases_change);
	free_matrix(cost_ddx_current_layer_activation);
}

void back_propagate_errors(struct Layer* l, float* expectations, float learn_rate) {
	if (!l->has_previous_layer) {
		return;
	}
	
	struct Matrix* cost_ddx_current_layer_activation = clone_matrix(*l->nodes);
	for (int i = 0; i < l->num_nodes; i++) {
		cost_ddx_current_layer_activation->data[i] = 2 * (cost_ddx_current_layer_activation->data[i] - expectations[i]);
	}

	struct Matrix* biases_change = clone_matrix(*l->raw_nodes);
	l->activation_ddx(biases_change->data, l->num_nodes);
	matrix_multiply_elementwise(biases_change, cost_ddx_current_layer_activation);
	matrix_scale(biases_change, -learn_rate);

	matrix_transpose(l->previous_layer->nodes);
	struct Matrix* weights_change = matrix_multiply(*biases_change, *l->previous_layer->nodes);
	matrix_transpose(l->previous_layer->nodes);

	do_back_propagate_errors(l->previous_layer, l, cost_ddx_current_layer_activation, learn_rate);

	matrix_add(l->weights, weights_change);
	matrix_add(l->biases, biases_change);

	free_matrix(weights_change);
	free_matrix(biases_change);
	free_matrix(cost_ddx_current_layer_activation);
}