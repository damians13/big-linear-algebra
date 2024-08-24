#include "mnist_csv.h"
#include <stdlib.h>
#include "csv.h"

const char* TRAINING_FILEPATH = "data/mnist/mnist_debug.csv";
const char* TESTING_FILEPATH = "data/mnist/mnist_test.csv";
float* training_data;
float* testing_data;
int training_data_index;
int testing_data_index;

void load_training_data() {
	training_data = read_csv_contents(TRAINING_FILEPATH);
	training_data_index = 0;
}

void load_testing_data() {
	testing_data = read_csv_contents(TESTING_FILEPATH);
	testing_data_index = 0;
}

float* get_next_training_data() {
	float* address = training_data + training_data_index * 784;
	training_data_index++;
	return address;
}

float* get_next_testing_data() {
	float* address = testing_data + testing_data_index * 784;
	testing_data_index++;
	return address;
}

void free_training_data() {
	free(training_data);
}

void free_testing_data() {
	free(testing_data);
}