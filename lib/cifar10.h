#ifndef __cifar10_h__
#define __cifar10_h__
#include <stdint.h>
#include <stdio.h>

extern const unsigned int CIFAR10_NUM_EXAMPLES_PER_FILE;
extern const unsigned int CIFAR10_LINE_LENGTH;
extern const unsigned int CIFAR10_DATA_LENGTH;
extern const unsigned int CIFAR10_BATCH_FILE_SIZE;
extern const unsigned int CIFAR10_NUM_PIXELS;
extern const unsigned int CIFAR10_EXAMPLE_DIM;

// Fill arr with 3072 bytes of pixel data (1024 red followed by 1024 green and 1024 blue)
void fill_random_data(int fd, uint8_t* arr);

#endif