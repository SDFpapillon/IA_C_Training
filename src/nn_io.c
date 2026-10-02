#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

#define NN_FILE_MAGIC "NNTXT1"

nn_status nn_save(const nn_network *net, const char *path)
{
    if (net == NULL || path == NULL) {
        return NN_ERR_NULL_ARG;
    }

    FILE *f = fopen(path, "w");
    if (f == NULL) {
        return NN_ERR_IO;
    }

    nn_status status = NN_OK;
    if (fprintf(f, "%s\n%zu\n", NN_FILE_MAGIC, net->n_layers) < 0) {
        status = NN_ERR_IO;
        goto done;
    }

    for (size_t i = 0; i < net->n_layers; i++) {
        const nn_layer *layer = &net->layers[i];
        size_t n_weights = layer->n_inputs * layer->n_outputs;

        if (fprintf(f, "%zu %zu %d\n", layer->n_inputs, layer->n_outputs,
                    (int)layer->activation) < 0) {
            status = NN_ERR_IO;
            goto done;
        }
        for (size_t w = 0; w < n_weights; w++) {
            char sep = (w + 1 < n_weights) ? ' ' : '\n';
            if (fprintf(f, "%.17g%c", layer->weights[w], sep) < 0) {
                status = NN_ERR_IO;
                goto done;
            }
        }
        for (size_t b = 0; b < layer->n_outputs; b++) {
            char sep = (b + 1 < layer->n_outputs) ? ' ' : '\n';
            if (fprintf(f, "%.17g%c", layer->biases[b], sep) < 0) {
                status = NN_ERR_IO;
                goto done;
            }
        }
    }

done:
    if (fclose(f) != 0 && status == NN_OK) {
        status = NN_ERR_IO;
    }
    return status;
}

nn_status nn_load(const char *path, nn_network **out)
{
    if (path == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return NN_ERR_IO;
    }

    nn_status status = NN_OK;
    nn_network *net = NULL;

    char magic[16] = {0};
    if (fscanf(f, "%15s", magic) != 1 || strcmp(magic, NN_FILE_MAGIC) != 0) {
        status = NN_ERR_FORMAT;
        goto done;
    }

    size_t n_layers;
    if (fscanf(f, "%zu", &n_layers) != 1 || n_layers == 0) {
        status = NN_ERR_FORMAT;
        goto done;
    }

    net = malloc(sizeof(*net));
    if (net == NULL) {
        status = NN_ERR_ALLOC;
        goto done;
    }
    net->n_layers = n_layers;
    net->layers = calloc(n_layers, sizeof(*net->layers));
    if (net->layers == NULL) {
        status = NN_ERR_ALLOC;
        goto done;
    }

    for (size_t i = 0; i < n_layers; i++) {
        size_t n_in, n_out;
        int act;
        if (fscanf(f, "%zu %zu %d", &n_in, &n_out, &act) != 3 ||
            n_in == 0 || n_out == 0 ||
            act < (int)NN_ACT_LINEAR || act > (int)NN_ACT_RELU) {
            status = NN_ERR_FORMAT;
            goto done;
        }

        nn_layer *layer = &net->layers[i];
        size_t n_weights = n_in * n_out;
        layer->n_inputs = n_in;
        layer->n_outputs = n_out;
        layer->activation = (nn_activation)act;
        layer->weights = malloc(n_weights * sizeof(*layer->weights));
        layer->biases = malloc(n_out * sizeof(*layer->biases));
        if (layer->weights == NULL || layer->biases == NULL) {
            status = NN_ERR_ALLOC;
            goto done;
        }

        for (size_t w = 0; w < n_weights; w++) {
            double v;
            if (fscanf(f, "%lf", &v) != 1) {
                status = NN_ERR_FORMAT;
                goto done;
            }
            layer->weights[w] = (nn_real)v;
        }
        for (size_t b = 0; b < n_out; b++) {
            double v;
            if (fscanf(f, "%lf", &v) != 1) {
                status = NN_ERR_FORMAT;
                goto done;
            }
            layer->biases[b] = (nn_real)v;
        }
    }

done:
    fclose(f);
    if (status != NN_OK) {
        nn_free(net);
        return status;
    }
    *out = net;
    return NN_OK;
}
