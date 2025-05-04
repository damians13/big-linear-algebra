#include "../lib/bmp.h"
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv) {
	unsigned int width = 1100;
	unsigned int height = 256;
	uint8_t* data = malloc(3 * width * height);
	uint8_t* red = data;
	uint8_t* green = data + width * height;
	uint8_t* blue = data + 2 * width * height;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			red[i * width + j] = 255;
			green[i * width + j] = i;
			blue[i * width + j] = (10 * j) % 256;
		}
	}

	BMPData bmp = { width, height, red, green, blue };

	write_bmp_data("test.bmp", &bmp);

	free(data);
}