#ifndef __matrix_h__
#define __matrix_h__

typedef double matrix_float_t;

// Data is stored in row major order
typedef struct Matrix {
	int rows;
	int cols;
	matrix_float_t* data;
} Matrix;

struct Matrix* make_matrix(int rows, int cols, matrix_float_t* data);
struct Matrix* clone_matrix(struct Matrix m);
void free_matrix_data(struct Matrix* m);
void free_matrix(struct Matrix* m);
struct Matrix* matrix_multiply(struct Matrix a, struct Matrix b);
void matrix_scale(struct Matrix* m, matrix_float_t f);
void matrix_add(struct Matrix* a, struct Matrix* b);
void print_matrix(struct Matrix m);
void print_matrix_dim(struct Matrix m);
void matrix_multiply_elementwise(struct Matrix* a, struct Matrix* b);
void matrix_transpose(struct Matrix* m);
struct Matrix* matrix_row_sum(struct Matrix m);
struct Matrix* matrix_col_sum(struct Matrix m);
matrix_float_t frobenius_norm(struct Matrix m);
matrix_float_t max_value(struct Matrix m);
void matrix_z_score_normalize(Matrix* m);
void matrix_add_tile_columns(struct Matrix* a, struct Matrix* b);

void matrix_multiply_inplace(Matrix* a, Matrix* b, Matrix* c);

#endif