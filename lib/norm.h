#ifndef __norm_h__
#define __norm_h__

#include "matrix.h"

void group_norm(Matrix* in, Matrix* out, matrix_float_t* stdevs, int channels, int group_size);

#endif