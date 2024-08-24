#include "csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

// Side-effect: seeks `f` to the end of the file
int get_num_values(FILE* f) {
	rewind(f);
	int count = 0;
	while (!feof(f)) {
		if (fgetc(f) == ',') {
			count++;
		}
	}
	return count;
}

float* read_csv_contents(const char* filepath) {
	FILE* f = fopen(filepath, "r");
	if (feof(f)) {
		printf("CSV file at %s is empty\n", filepath);
		return 0;
	}

	int numValues = get_num_values(f); // Seeks f to end of file
	long numChars = ftell(f);
	rewind(f);
	
	float* values = malloc(numValues * sizeof(float));
	int digitCount = 0;

	char digitString[4 + DBL_MANT_DIG - DBL_MIN_EXP];
	int charCount = 0;

	char c = '\0';
	while (!feof(f)) {
		c = fgetc(f);
		if (c == ',' || (c == '\n' && charCount != 0)) {
			digitString[charCount] = '\0';
			values[digitCount] = atof(digitString);
			charCount = 0;
			digitCount++;
		} else if (c != '\n') {
			digitString[charCount] = c;
			charCount++;
		}
	}

	fclose(f);
	return values;
}

void write_csv_contents(const char* filepath, float* data, int cols, int rows) {
	FILE* f = fopen(filepath, "w");
	for (int i = 0; i < cols * rows; i++) {
		fprintf(f, "%f,", data[i]);
		if ((i + 1) % cols == 0) {
			fputs("\n", f);
		}
	}
	fflush(f);
	fclose(f);
}