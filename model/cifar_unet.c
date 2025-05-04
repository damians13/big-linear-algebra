#include "../lib/bmp.h"
#include "../lib/cifar10.h"
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>

int main(int argc, char** argv) {
	uint8_t* data = malloc(CIFAR10_DATA_LENGTH);

	int fd = open("data/cifar/data_batch_1.bin", O_RDONLY);

	srand(43);
	fill_random_data(fd, data);

	uint8_t* red = data;
	uint8_t* green = data + CIFAR10_NUM_PIXELS;
	uint8_t* blue = data + 2 * CIFAR10_NUM_PIXELS;

	BMPData bmp = { 32, 32, red, green, blue };

	write_bmp_data("test.bmp", &bmp);

	free(data);
}