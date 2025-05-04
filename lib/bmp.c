#include "bmp.h"
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

// All integers are little-endian
// https://en.wikipedia.org/wiki/BMP_file_format
void write_bmp_data(const char* filepath, BMPData* data) {
	int fd = open(filepath, O_WRONLY | O_CREAT, 0777);
	unsigned int pixel_data_row_size = ((24 * data->width + 31) / 32) * 4;
	unsigned int file_size = 54 + pixel_data_row_size * data->height;

	uint8_t bitmap_header[14];
	bitmap_header[0] = 'B'; // Indicate start of BMP file
	bitmap_header[1] = 'M'; // Indicate start of BMP file
	bitmap_header[2] = file_size & 0xFF; // 4 bytes for overall file size
	bitmap_header[3] = (file_size >> 8) & 0xFF; // 4 bytes for overall file size
	bitmap_header[4] = (file_size >> 16) & 0xFF; // 4 bytes for overall file size
	bitmap_header[5] = (file_size >> 24) & 0xFF; // 4 bytes for overall file size
	bitmap_header[6] = 0; // Reserved
	bitmap_header[7] = 0; // Reserved
	bitmap_header[8] = 0; // Reserved
	bitmap_header[9] = 0; // Reserved
	bitmap_header[10] = 54; // 4 bytes for the offset of the first pixel data in the file
	bitmap_header[11] =  0; // 4 bytes for the offset of the first pixel data in the file
	bitmap_header[12] =  0; // 4 bytes for the offset of the first pixel data in the file
	bitmap_header[13] =  0; // 4 bytes for the offset of the first pixel data in the file

	if (write(fd, (void*) bitmap_header, 14) != 14) {
		fprintf(stderr, "Error while writing bitmap header (errno=%d).\n", errno);
	}

	uint8_t bitmap_info_header[40];
	bitmap_info_header[0] = 40; // 4 bytes for header size
	bitmap_info_header[1] =  0; // 4 bytes for header size
	bitmap_info_header[2] =  0; // 4 bytes for header size
	bitmap_info_header[3] =  0; // 4 bytes for header size
	bitmap_info_header[4] = data->width & 0xFF; // 4 bytes for image width
	bitmap_info_header[5] = (data->width >> 8) & 0xFF; // 4 bytes for image width
	bitmap_info_header[6] = (data->width >> 16) & 0xFF; // 4 bytes for image width
	bitmap_info_header[7] = (data->width >> 24) & 0x7F; // 4 bytes for image width (signed)
	bitmap_info_header[8] = data->height & 0xFF; // 4 bytes for image height
	bitmap_info_header[9] = (data->height >> 8) & 0xFF; // 4 bytes for image height
	bitmap_info_header[10] = (data->height >> 16) & 0xFF; // 4 bytes for image height
	bitmap_info_header[11] = (data->height >> 24) & 0x7F; // 4 bytes for image height (signed)
	bitmap_info_header[12] = 1; // 2 bytes for number of colour planes
	bitmap_info_header[13] = 0; // 2 bytes for number of colour planes
	bitmap_info_header[14] = 24; // 2 bytes for number of bits per pixel (8 per channel)
	bitmap_info_header[15] = 0; // 2 bytes for number of bits per pixel (8 per channel)
	bitmap_info_header[16] = 0; // 4 bytes for compression method (none)
	bitmap_info_header[17] = 0; // 4 bytes for compression method (none)
	bitmap_info_header[18] = 0; // 4 bytes for compression method (none)
	bitmap_info_header[19] = 0; // 4 bytes for compression method (none)
	bitmap_info_header[20] = 0; // 4 bytes for raw image size (dummy 0's work fine)
	bitmap_info_header[21] = 0; // 4 bytes for raw image size (dummy 0's work fine)
	bitmap_info_header[22] = 0; // 4 bytes for raw image size (dummy 0's work fine)
	bitmap_info_header[23] = 0; // 4 bytes for raw image size (dummy 0's work fine)
	bitmap_info_header[24] = 72; // 4 bytes for horizontal resolution
	bitmap_info_header[25] =  0; // 4 bytes for horizontal resolution
	bitmap_info_header[26] =  0; // 4 bytes for horizontal resolution
	bitmap_info_header[27] =  0; // 4 bytes for horizontal resolution
	bitmap_info_header[28] = 72; // 4 bytes for vertical resolution
	bitmap_info_header[29] =  0; // 4 bytes for vertical resolution
	bitmap_info_header[30] =  0; // 4 bytes for vertical resolution
	bitmap_info_header[31] =  0; // 4 bytes for vertical resolution
	bitmap_info_header[32] = 0; // 4 bytes for number of unique colours in the image
	bitmap_info_header[32] = 1; // 4 bytes for number of unique colours in the image
	bitmap_info_header[34] = 0; // 4 bytes for number of unique colours in the image
	bitmap_info_header[35] = 0; // 4 bytes for number of unique colours in the image
	bitmap_info_header[36] = 0; // 4 bytes for number of important colours in the image
	bitmap_info_header[37] = 0; // 4 bytes for number of important colours in the image
	bitmap_info_header[38] = 0; // 4 bytes for number of important colours in the image
	bitmap_info_header[39] = 0; // 4 bytes for number of important colours in the image

	if (write(fd, (void*) bitmap_info_header, 40) != 40) {
		fprintf(stderr, "Error while writing bitmap info header (errno=%d).\n", errno);
	}

	uint8_t* pixel_data = malloc(pixel_data_row_size * data->height);
	for (int i = 0; i < data->height; i++) {
		for (int j = 0; j < data->width; j++) {
			pixel_data[i * pixel_data_row_size + 3 * j] = data->blue[i * data->width + j];
			pixel_data[i * pixel_data_row_size + 3 * j + 1] = data->green[i * data->width + j];
			pixel_data[i * pixel_data_row_size + 3 * j + 2] = data->red[i * data->width + j];
		}
		// Pad the end of the row with any necessary 0's
		for (int j = 3 * data->width; j < pixel_data_row_size; j++) {
			pixel_data[i * pixel_data_row_size + j] = 0;
		}
	}

	if (write(fd, (void*) pixel_data, pixel_data_row_size * data->height) != pixel_data_row_size * data->height) {
		fprintf(stderr, "Error while writing bitmap pixel data (errno=%d).\n", errno);
	}

	free(pixel_data);
	close(fd);
}