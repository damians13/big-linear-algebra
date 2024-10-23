#ifndef __csv_h__
#define __csv_h__

#include <stdio.h>

float* read_csv_contents(const char* filepath);
void write_csv_contents(const char* filepath, float* data, int cols, int rows);
int count_num_lines(FILE* f);

#endif