#ifndef __bmp_h__
#define __bmp_h__

#include <stdint.h>

typedef struct BMPData {
	unsigned int width;
	unsigned int height;
	uint8_t* red;
	uint8_t* green;
	uint8_t* blue;
} BMPData;

void write_bmp_data(const char* filepath, BMPData* data);

#endif