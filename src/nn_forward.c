#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"
#include "nn_internal.h"

nn_status nn_forward_buffer_create(const nn_network *net, nn_forward_buffer **out)
{
    if (net == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_forward_buffer *buf = malloc(sizeof(*buf));
    if (buf == NULL) {
        return NN_ERR_ALLOC;
    }
    buf->n_layers = net->n_layers;
    buf->activations = calloc(net->n_layers, sizeof(*buf->activations));
    if (buf->activations == NULL) {
        free(buf);
        return NN_ERR_ALLOC;
    }

    for (size_t l = 0; l < net->n_layers; l++) {
        buf->activations[l] = malloc(net->layers[l].n_outputs * sizeof(*buf->activations[l]));
        if (buf->activations[l] == NULL) {
            nn_forward_buffer_free(buf);
            return NN_ERR_ALLOC;
        }
    }

    *out = buf;
    return NN_OK;
}

void nn_forward_buffer_free(nn_forward_buffer *buf)
{
    if (buf == NULL) {
        return;
    }
    for (size_t l = 0; l < buf->n_layers; l++) {
        free(buf->activations[l]);
    }
    free(buf->activations);
    free(buf);
}

void nn_softmax_inplace(nn_real *values, size_t n)
{
    nn_real max_val = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }

    nn_real sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        values[i] = exp((double)(values[i] - max_val));
        sum += values[i];
    }
    for (size_t i = 0; i < n; i++) {
        values[i] /= sum;
    }
}

nn_status nn_forward(const nn_network *net, nn_forward_buffer *buf,
                      const nn_real *input, nn_real *output)
{
    if (net == NULL || buf == NULL || input == NULL || output == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (net->n_layers != buf->n_layers) {
        return NN_ERR_SIZE_MISMATCH;
    }

    const nn_real *layer_input = input;

    for (size_t l = 0; l < net->n_layers; l++) {
        const nn_layer *layer = &net->layers[l];
        nn_real *layer_output = buf->activations[l];

        for (size_t o = 0; o < layer->n_outputs; o++) {
            const nn_real *row = &layer->weights[o * layer->n_inputs];
            nn_real sum = layer->biases[o];
            for (size_t i = 0; i < layer->n_inputs; i++) {
                sum += row[i] * layer_input[i];
            }
            layer_output[o] = (layer->activation == NN_ACT_SOFTMAX)
                                   ? sum
                                   : nn_activation_apply(layer->activation, sum);
        }

        if (layer->activation == NN_ACT_SOFTMAX) {
            nn_softmax_inplace(layer_output, layer->n_outputs);
        }

        layer_input = layer_output;
    }

    size_t n_out = net->layers[net->n_layers - 1].n_outputs;
    memcpy(output, buf->activations[net->n_layers - 1], n_out * sizeof(*output));
    return NN_OK;
}
