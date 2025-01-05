#include "mnist_csv2.h"
#include "csv.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h> // memset

#define MNIST_CSV_LINE_LENGTH 785

/**
 * csv->file should be set but anything else will be overridden
 */
void mnist_csv_init(MnistCSV* csv) {
	float* csv_contents = read_csv_contents_file(csv->file, &csv->num_examples);
	printf("MNIST CSV file contents read!\n");
	csv->num_examples /= MNIST_CSV_LINE_LENGTH;

	csv->X = malloc(csv->num_examples * (MNIST_CSV_LINE_LENGTH - 1) * sizeof(float));
	csv->y = malloc(csv->num_examples * sizeof(float));
	csv->num_sampled = 0;
	csv->sampled = malloc(csv->num_examples * sizeof(char));
	
	memset(csv->sampled, 0, csv->num_examples);

	// Split csv_contents into X and y
	for (int i = 0; i < csv->num_examples; i++) {
		csv->y[i] = csv_contents[i * MNIST_CSV_LINE_LENGTH];
		for (int j = 0; j < MNIST_CSV_LINE_LENGTH - 1; j++) {
			csv->X[i + j * csv->num_examples] = csv_contents[i * MNIST_CSV_LINE_LENGTH + j + 1];
		}
	}

	free(csv_contents);
}

MnistExample get_random_data_replace(MnistCSV* csv) {
	int n = floor((float)csv->num_examples * (float)rand() / (float)RAND_MAX);
	return (MnistExample) { csv->X + n, csv->y[n], csv->num_examples };
}

MnistExample get_random_data_take(MnistCSV* csv) {
	// Reset sampled if all have been sampled (ie. start over)
	if (csv->num_sampled == csv->num_examples) {
		csv->num_sampled = 0;
		memset(csv->sampled, 0, csv->num_examples);
	}

	// Pick a random line from the file that hasn't been sampled yet (the `n`th unsampled line)
	int n = floor((float)(csv->num_examples - csv->num_sampled) * (float)rand() / (float)RAND_MAX);

	// Get to that data point by skipping any previously sampled points as well as `n` unsampled points
	int i;
	for (i = 0; i < csv->num_examples && n > 0; i++) {
		if (csv->sampled[i] == 0) {
			n--;
		}
	}
	csv->sampled[i] = 1;
	csv->num_sampled++;

	return (MnistExample) { csv->X + i, csv->y[i], csv->num_examples };
}

void visualize_digit_data(MnistExample ex) {
	printf("============================\n");
	printf("Data for digit %f:\n", ex.y);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (ex.X[j * ex.num_examples + i * 28 * ex.num_examples] < 80) {
				printf(" ");
			} else if (ex.X[j * ex.num_examples + i * 28 * ex.num_examples] < 150) {
				printf(":");
			} else {
				printf("#");
			}
		}
		printf("\n");
	}
	printf("============================\n");
}