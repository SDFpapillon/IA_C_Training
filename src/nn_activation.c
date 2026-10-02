#include <math.h>

#include "nn.h"

static nn_real nn_sigmoid(nn_real x)
{
    return 1.0 / (1.0 + exp((double)-x));
}

nn_real nn_activation_apply(nn_activation act, nn_real x)
{
    switch (act) {
    case NN_ACT_LINEAR:
        return x;
    case NN_ACT_SIGMOID:
        return nn_sigmoid(x);
    case NN_ACT_TANH:
        return tanh((double)x);
    case NN_ACT_RELU:
        return x > 0.0 ? x : 0.0;
    }
    return x;
}

nn_real nn_activation_derivative(nn_activation act, nn_real x)
{
    switch (act) {
    case NN_ACT_LINEAR:
        return 1.0;
    case NN_ACT_SIGMOID: {
        nn_real s = nn_sigmoid(x);
        return s * (1.0 - s);
    }
    case NN_ACT_TANH: {
        nn_real t = tanh((double)x);
        return 1.0 - t * t;
    }
    case NN_ACT_RELU:
        return x > 0.0 ? 1.0 : 0.0;
    }
    return 1.0;
}
