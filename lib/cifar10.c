#include "cifar10.h"
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

const unsigned int CIFAR10_NUM_EXAMPLES_PER_FILE = 10000;
const unsigned int CIFAR10_LINE_LENGTH = 3073;
const unsigned int CIFAR10_DATA_LENGTH = 3072;
const unsigned int CIFAR10_BATCH_FILE_SIZE = 30730000;
const unsigned int CIFAR10_NUM_PIXELS = 1024;
const unsigned int CIFAR10_EXAMPLE_DIM = 32;

void fill_random_data(int fd, uint8_t* arr) {
	unsigned int example = ((float) rand() / ((float) RAND_MAX + 1)) * CIFAR10_NUM_EXAMPLES_PER_FILE;
	if (lseek(fd, example * CIFAR10_LINE_LENGTH + 1, SEEK_SET) != example * CIFAR10_LINE_LENGTH + 1) {
		fprintf(stderr, "Error while seeking to CIFAR10 example %d (errno=%d).\n", example, errno);
	}
	
	// CIFAR10 images are top down but BMP expects bottom up, so flip them for nicer previews
	uint8_t buffer[CIFAR10_DATA_LENGTH];
	if (read(fd, buffer, CIFAR10_DATA_LENGTH) != CIFAR10_DATA_LENGTH) {
		fprintf(stderr, "Error while reading CIFAR10 example %d (errno=%d).\n", example, errno);
	}
	for (int i = 0; i < CIFAR10_EXAMPLE_DIM; i++) {
		for (int j = 0; j < CIFAR10_EXAMPLE_DIM; j++) {
			unsigned int reversed_i = CIFAR10_EXAMPLE_DIM - i - 1;
			arr[i * CIFAR10_EXAMPLE_DIM + j] = buffer[reversed_i * CIFAR10_EXAMPLE_DIM + j]; // Red
			arr[i * CIFAR10_EXAMPLE_DIM + j + CIFAR10_NUM_PIXELS] = buffer[reversed_i * CIFAR10_EXAMPLE_DIM + j + CIFAR10_NUM_PIXELS]; // Green
			arr[i * CIFAR10_EXAMPLE_DIM + j + 2 * CIFAR10_NUM_PIXELS] = buffer[reversed_i * CIFAR10_EXAMPLE_DIM + j + 2 * CIFAR10_NUM_PIXELS]; // Blue
		}
	}
}