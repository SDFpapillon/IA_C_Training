#include "nn.h"

size_t nn_genome_size(const nn_network *net)
{
    size_t total = 0;
    for (size_t l = 0; l < net->n_layers; l++) {
        total += net->layers[l].n_inputs * net->layers[l].n_outputs + net->layers[l].n_outputs;
    }
    return total;
}

void nn_genome_flatten(const nn_network *net, nn_real *genome)
{
    size_t pos = 0;
    for (size_t l = 0; l < net->n_layers; l++) {
        const nn_layer *layer = &net->layers[l];
        size_t n_w = layer->n_inputs * layer->n_outputs;
        for (size_t w = 0; w < n_w; w++) {
            genome[pos++] = layer->weights[w];
        }
        for (size_t o = 0; o < layer->n_outputs; o++) {
            genome[pos++] = layer->biases[o];
        }
    }
}

void nn_genome_unflatten(nn_network *net, const nn_real *genome)
{
    size_t pos = 0;
    for (size_t l = 0; l < net->n_layers; l++) {
        nn_layer *layer = &net->layers[l];
        size_t n_w = layer->n_inputs * layer->n_outputs;
        for (size_t w = 0; w < n_w; w++) {
            layer->weights[w] = genome[pos++];
        }
        for (size_t o = 0; o < layer->n_outputs; o++) {
            layer->biases[o] = genome[pos++];
        }
    }
}
