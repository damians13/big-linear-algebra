#include <stdio.h>
#include "../lib/mnist_csv.h"

int main(int argc, char** argv) {
	load_training_data();
	float* training = get_next_training_data();

	printf("Training data for digit %.f:\n", training[0]);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (training[i * 28 + j + 1] < 127) {
				printf(" ");
			} else {
				printf("#");
			}
		}
		printf("\n");
	}

	free_training_data();
}