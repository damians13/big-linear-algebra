#include "mnist_csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int get_next_data(struct MnistCSV* csv) {
	if (feof(csv->file)) {
		printf("CSV file is empty\n");
		return 1;
	}

	int index = 0;
	int charCount = 0;
	char digitString[4];
	while (index < 785) {
		char c = fgetc(csv->file);
		if (c == ',' || (c == '\n' && charCount != 0)) {
			digitString[charCount] = '\0';
			csv->buffer[index] = atof(digitString);
			charCount = 0;
			index++;
		} else if (c != '\n') {
			digitString[charCount] = c;
			charCount++;
		}
	}

	return 0;
}

void visualize_digit_data(struct MnistCSV* csv) {
	float* digit = csv->buffer;
	printf("============================\n");
	printf("Data for digit %.f:\n", digit[0]);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (digit[i * 28 + j + 1] < 0.32) {
				printf(" ");
			} else if (digit[i * 28 + j + 1] < 0.6) {
				printf(":");
			} else {
				printf("#");
			}
		}
		printf("\n");
	}
	printf("============================\n");
}