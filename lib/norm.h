#ifndef __norm_h__
#define __norm_h__

#include "matrix.h"

void group_norm(Matrix* in, Matrix* out, matrix_float_t* stdevs, matrix_float_t* means, int channels, int group_size);
void group_norm_ddx(Matrix* source, Matrix* dest, Matrix* data, matrix_float_t* means, matrix_float_t* stdevs, int channels, int group_size);

#endif