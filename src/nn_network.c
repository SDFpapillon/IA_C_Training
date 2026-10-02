#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

static void nn_layer_free_contents(nn_layer *layer)
{
    free(layer->weights);
    free(layer->biases);
}

void nn_free(nn_network *net)
{
    if (net == NULL) {
        return;
    }
    for (size_t i = 0; i < net->n_layers; i++) {
        nn_layer_free_contents(&net->layers[i]);
    }
    free(net->layers);
    free(net);
}

/* Half-width of the uniform init range, given the strategy and layer shape. */
static nn_real nn_init_limit(nn_init init, size_t fan_in, size_t fan_out)
{
    switch (init) {
    case NN_INIT_XAVIER:
        return sqrt(6.0 / (double)(fan_in + fan_out));
    case NN_INIT_HE:
        return sqrt(6.0 / (double)fan_in);
    case NN_INIT_UNIFORM:
    default:
        return 1.0;
    }
}

nn_status nn_create(const size_t *layer_sizes, size_t n_layer_sizes,
                     const nn_activation *activations, size_t n_activations,
                     nn_init init, uint64_t seed, nn_network **out)
{
    if (layer_sizes == NULL || activations == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (n_layer_sizes < 2 || n_activations != n_layer_sizes - 1) {
        return NN_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < n_layer_sizes; i++) {
        if (layer_sizes[i] == 0) {
            return NN_ERR_INVALID_ARG;
        }
    }

    size_t n_layers = n_layer_sizes - 1;

    nn_network *net = malloc(sizeof(*net));
    if (net == NULL) {
        return NN_ERR_ALLOC;
    }
    net->n_layers = n_layers;
    net->layers = calloc(n_layers, sizeof(*net->layers));
    if (net->layers == NULL) {
        free(net);
        return NN_ERR_ALLOC;
    }

    nn_rng rng;
    nn_rng_seed(&rng, seed);

    for (size_t i = 0; i < n_layers; i++) {
        size_t n_in = layer_sizes[i];
        size_t n_out = layer_sizes[i + 1];
        size_t n_weights = n_in * n_out;
        nn_layer *layer = &net->layers[i];

        layer->n_inputs = n_in;
        layer->n_outputs = n_out;
        layer->activation = activations[i];
        layer->weights = malloc(n_weights * sizeof(*layer->weights));
        layer->biases = calloc(n_out, sizeof(*layer->biases));
        if (layer->weights == NULL || layer->biases == NULL) {
            nn_free(net);
            return NN_ERR_ALLOC;
        }

        nn_real limit = nn_init_limit(init, n_in, n_out);
        for (size_t w = 0; w < n_weights; w++) {
            layer->weights[w] = nn_rng_uniform_range(&rng, -limit, limit);
        }
        /* biases already zero-initialized by calloc */
    }

    *out = net;
    return NN_OK;
}

nn_status nn_clone(const nn_network *net, nn_network **out)
{
    if (net == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_network *copy = malloc(sizeof(*copy));
    if (copy == NULL) {
        return NN_ERR_ALLOC;
    }
    copy->n_layers = net->n_layers;
    copy->layers = calloc(net->n_layers, sizeof(*copy->layers));
    if (copy->layers == NULL) {
        free(copy);
        return NN_ERR_ALLOC;
    }

    for (size_t i = 0; i < net->n_layers; i++) {
        const nn_layer *src = &net->layers[i];
        nn_layer *dst = &copy->layers[i];
        size_t n_weights = src->n_inputs * src->n_outputs;

        dst->n_inputs = src->n_inputs;
        dst->n_outputs = src->n_outputs;
        dst->activation = src->activation;
        dst->weights = malloc(n_weights * sizeof(*dst->weights));
        dst->biases = malloc(src->n_outputs * sizeof(*dst->biases));
        if (dst->weights == NULL || dst->biases == NULL) {
            nn_free(copy);
            return NN_ERR_ALLOC;
        }
        memcpy(dst->weights, src->weights, n_weights * sizeof(*dst->weights));
        memcpy(dst->biases, src->biases, src->n_outputs * sizeof(*dst->biases));
    }

    *out = copy;
    return NN_OK;
}
