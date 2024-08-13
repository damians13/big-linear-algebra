#ifndef __layer_h__
#define __layer_h__

struct Layer {
	int num_nodes;
	struct Matrix* nodes;
	struct Matrix* weights;
	struct Matrix* biases;
	struct Layer* previous_layer;
	void (*activation)(float*, int);
	char has_previous_layer;
	char has_nodes;
};

void feed_forward(struct Layer* l);
void free_layer_data(struct Layer l);
void load_weights_from_csv(struct Layer* l, const char* filepath);
void load_biases_from_csv(struct Layer* l, const char* filepath);

#endif