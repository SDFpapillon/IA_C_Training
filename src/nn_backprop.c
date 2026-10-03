#include <stdlib.h>
#include <string.h>

#include "nn.h"
#include "nn_internal.h"

nn_status nn_gradient_create(const nn_network *net, nn_gradient **out)
{
    if (net == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_gradient *grad = malloc(sizeof(*grad));
    if (grad == NULL) {
        return NN_ERR_ALLOC;
    }
    grad->n_layers = net->n_layers;
    grad->n_inputs = malloc(net->n_layers * sizeof(*grad->n_inputs));
    grad->n_outputs = malloc(net->n_layers * sizeof(*grad->n_outputs));
    grad->dweights = calloc(net->n_layers, sizeof(*grad->dweights));
    grad->dbiases = calloc(net->n_layers, sizeof(*grad->dbiases));
    if (grad->n_inputs == NULL || grad->n_outputs == NULL ||
        grad->dweights == NULL || grad->dbiases == NULL) {
        nn_gradient_free(grad);
        return NN_ERR_ALLOC;
    }

    for (size_t l = 0; l < net->n_layers; l++) {
        size_t n_in = net->layers[l].n_inputs;
        size_t n_out = net->layers[l].n_outputs;
        grad->n_inputs[l] = n_in;
        grad->n_outputs[l] = n_out;
        grad->dweights[l] = malloc(n_in * n_out * sizeof(*grad->dweights[l]));
        grad->dbiases[l] = malloc(n_out * sizeof(*grad->dbiases[l]));
        if (grad->dweights[l] == NULL || grad->dbiases[l] == NULL) {
            nn_gradient_free(grad);
            return NN_ERR_ALLOC;
        }
    }

    nn_gradient_zero(grad);
    *out = grad;
    return NN_OK;
}

void nn_gradient_free(nn_gradient *grad)
{
    if (grad == NULL) {
        return;
    }
    if (grad->dweights != NULL) {
        for (size_t l = 0; l < grad->n_layers; l++) {
            free(grad->dweights[l]);
        }
    }
    if (grad->dbiases != NULL) {
        for (size_t l = 0; l < grad->n_layers; l++) {
            free(grad->dbiases[l]);
        }
    }
    free(grad->dweights);
    free(grad->dbiases);
    free(grad->n_inputs);
    free(grad->n_outputs);
    free(grad);
}

void nn_gradient_zero(nn_gradient *grad)
{
    for (size_t l = 0; l < grad->n_layers; l++) {
        size_t n_w = grad->n_inputs[l] * grad->n_outputs[l];
        memset(grad->dweights[l], 0, n_w * sizeof(*grad->dweights[l]));
        memset(grad->dbiases[l], 0, grad->n_outputs[l] * sizeof(*grad->dbiases[l]));
    }
}

void nn_gradient_apply(nn_network *net, const nn_gradient *grad, nn_real learning_rate, size_t n_samples)
{
    nn_real scale = learning_rate / (nn_real)n_samples;

    for (size_t l = 0; l < net->n_layers; l++) {
        nn_layer *layer = &net->layers[l];
        size_t n_w = layer->n_inputs * layer->n_outputs;

        for (size_t w = 0; w < n_w; w++) {
            layer->weights[w] -= scale * grad->dweights[l][w];
        }
        for (size_t o = 0; o < layer->n_outputs; o++) {
            layer->biases[o] -= scale * grad->dbiases[l][o];
        }
    }
}

nn_status nn_backprop_buffer_create(const nn_network *net, nn_backprop_buffer **out)
{
    if (net == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_backprop_buffer *buf = malloc(sizeof(*buf));
    if (buf == NULL) {
        return NN_ERR_ALLOC;
    }
    buf->n_layers = net->n_layers;
    buf->z = calloc(net->n_layers, sizeof(*buf->z));
    buf->a = calloc(net->n_layers, sizeof(*buf->a));
    buf->delta = calloc(net->n_layers, sizeof(*buf->delta));
    if (buf->z == NULL || buf->a == NULL || buf->delta == NULL) {
        nn_backprop_buffer_free(buf);
        return NN_ERR_ALLOC;
    }

    for (size_t l = 0; l < net->n_layers; l++) {
        size_t n_out = net->layers[l].n_outputs;
        buf->z[l] = malloc(n_out * sizeof(*buf->z[l]));
        buf->a[l] = malloc(n_out * sizeof(*buf->a[l]));
        buf->delta[l] = malloc(n_out * sizeof(*buf->delta[l]));
        if (buf->z[l] == NULL || buf->a[l] == NULL || buf->delta[l] == NULL) {
            nn_backprop_buffer_free(buf);
            return NN_ERR_ALLOC;
        }
    }

    *out = buf;
    return NN_OK;
}

void nn_backprop_buffer_free(nn_backprop_buffer *buf)
{
    if (buf == NULL) {
        return;
    }
    if (buf->z != NULL) {
        for (size_t l = 0; l < buf->n_layers; l++) {
            free(buf->z[l]);
        }
    }
    if (buf->a != NULL) {
        for (size_t l = 0; l < buf->n_layers; l++) {
            free(buf->a[l]);
        }
    }
    if (buf->delta != NULL) {
        for (size_t l = 0; l < buf->n_layers; l++) {
            free(buf->delta[l]);
        }
    }
    free(buf->z);
    free(buf->a);
    free(buf->delta);
    free(buf);
}

nn_status nn_backprop(const nn_network *net, nn_backprop_buffer *buf,
                       const nn_real *input, const nn_real *target,
                       nn_loss loss, nn_gradient *grad)
{
    if (net == NULL || buf == NULL || input == NULL || target == NULL || grad == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (net->n_layers != buf->n_layers || net->n_layers != grad->n_layers) {
        return NN_ERR_SIZE_MISMATCH;
    }

    size_t last = net->n_layers - 1;
    const nn_layer *output_layer = &net->layers[last];
    if (loss == NN_LOSS_CROSS_ENTROPY && output_layer->activation != NN_ACT_SOFTMAX) {
        return NN_ERR_INVALID_ARG;
    }
    if (loss == NN_LOSS_MSE && output_layer->activation == NN_ACT_SOFTMAX) {
        return NN_ERR_INVALID_ARG;
    }

    /* Forward pass, caching pre-activation (z) and post-activation (a) per layer. */
    const nn_real *layer_input = input;
    for (size_t l = 0; l < net->n_layers; l++) {
        const nn_layer *layer = &net->layers[l];
        nn_real *z = buf->z[l];
        nn_real *a = buf->a[l];

        for (size_t o = 0; o < layer->n_outputs; o++) {
            const nn_real *row = &layer->weights[o * layer->n_inputs];
            nn_real sum = layer->biases[o];
            for (size_t i = 0; i < layer->n_inputs; i++) {
                sum += row[i] * layer_input[i];
            }
            z[o] = sum;
            a[o] = (layer->activation == NN_ACT_SOFTMAX) ? sum : nn_activation_apply(layer->activation, sum);
        }
        if (layer->activation == NN_ACT_SOFTMAX) {
            nn_softmax_inplace(a, layer->n_outputs);
        }

        layer_input = a;
    }

    /* Backward pass: start from the output layer's delta (dL/dz). */
    nn_real *output = buf->a[last];
    nn_real *delta_last = buf->delta[last];

    if (output_layer->activation == NN_ACT_SOFTMAX) {
        /* Combined softmax + categorical cross-entropy gradient simplifies to a - t. */
        for (size_t o = 0; o < output_layer->n_outputs; o++) {
            delta_last[o] = output[o] - target[o];
        }
    } else {
        nn_real n_out = (nn_real)output_layer->n_outputs;
        for (size_t o = 0; o < output_layer->n_outputs; o++) {
            nn_real dloss_da = (2.0 / n_out) * (output[o] - target[o]);
            delta_last[o] = dloss_da * nn_activation_derivative(output_layer->activation, buf->z[last][o]);
        }
    }

    for (size_t l = net->n_layers; l-- > 0;) {
        const nn_layer *layer = &net->layers[l];
        const nn_real *delta = buf->delta[l];
        const nn_real *prev_a = (l == 0) ? input : buf->a[l - 1];

        for (size_t o = 0; o < layer->n_outputs; o++) {
            grad->dbiases[l][o] += delta[o];
            nn_real *dw_row = &grad->dweights[l][o * layer->n_inputs];
            for (size_t i = 0; i < layer->n_inputs; i++) {
                dw_row[i] += delta[o] * prev_a[i];
            }
        }

        if (l > 0) {
            const nn_layer *prev_layer = &net->layers[l - 1];
            nn_real *delta_prev = buf->delta[l - 1];
            for (size_t i = 0; i < prev_layer->n_outputs; i++) {
                nn_real dloss_da = 0.0;
                for (size_t o = 0; o < layer->n_outputs; o++) {
                    dloss_da += delta[o] * layer->weights[o * layer->n_inputs + i];
                }
                delta_prev[i] = dloss_da * nn_activation_derivative(prev_layer->activation, buf->z[l - 1][i]);
            }
        }
    }

    return NN_OK;
}
