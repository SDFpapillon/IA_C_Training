#ifndef NN_INTERNAL_H
#define NN_INTERNAL_H

#include "nn.h"

/*
 * In-place softmax over a layer's pre-activation values. Shared between
 * nn_forward() and nn_backprop(), which both apply it to the output layer
 * when its activation is NN_ACT_SOFTMAX.
 */
void nn_softmax_inplace(nn_real *values, size_t n);

#endif /* NN_INTERNAL_H */
