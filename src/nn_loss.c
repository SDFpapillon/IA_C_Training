#include <math.h>

#include "nn.h"

nn_real nn_loss_value(nn_loss loss, const nn_real *output, const nn_real *target, size_t n)
{
    switch (loss) {
    case NN_LOSS_MSE: {
        nn_real sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            nn_real diff = output[i] - target[i];
            sum += diff * diff;
        }
        return sum / (nn_real)n;
    }
    case NN_LOSS_CROSS_ENTROPY: {
        const nn_real eps = 1e-12;
        nn_real sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            nn_real p = output[i] > eps ? output[i] : eps;
            sum += target[i] * log((double)p);
        }
        return -sum;
    }
    }
    return 0.0;
}
