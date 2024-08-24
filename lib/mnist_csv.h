#ifndef __mnist_csv_h__
#define __mnist_csv_h__

void load_training_data();
void load_testing_data();
// Returns a float array where the 1xt entry is the numerical value and the remaining 784 are pixel values 0-255
float* get_next_training_data();
// Returns a float array where the 1xt entry is the numerical value and the remaining 784 are pixel values 0-255
float* get_next_testing_data();
void free_training_data();
void free_testing_data();

#endif