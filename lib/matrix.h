#ifndef __matrix_h__
#define __matrix_h__

struct Matrix {
	int rows;
	int cols;
	float* data;
};

struct Matrix* make_matrix(int rows, int cols, float* data);
struct Matrix* clone_matrix(struct Matrix m);
void free_matrix_data(struct Matrix* m);
void free_matrix(struct Matrix* m);
struct Matrix* matrix_multiply(struct Matrix a, struct Matrix b);
void matrix_scale(struct Matrix* m, float f);
void matrix_add(struct Matrix* a, struct Matrix* b);
void print_matrix(struct Matrix m);
void matrix_multiply_elementwise(struct Matrix* a, struct Matrix* b);

#endif