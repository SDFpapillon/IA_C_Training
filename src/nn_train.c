#include <stdlib.h>

#include "nn.h"

/* Fisher-Yates shuffle of order[0..n), using rng. */
static void nn_shuffle_indices(size_t *order, size_t n, nn_rng *rng)
{
    for (size_t i = n; i-- > 1;) {
        size_t j = (size_t)(nn_rng_next_u64(rng) % (uint64_t)(i + 1));
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

nn_status nn_train_supervised(nn_network *net, const nn_dataset *dataset,
                               const nn_train_params *params)
{
    if (net == NULL || dataset == NULL || params == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (dataset->n_inputs != net->layers[0].n_inputs ||
        dataset->n_outputs != net->layers[net->n_layers - 1].n_outputs) {
        return NN_ERR_SIZE_MISMATCH;
    }

    size_t batch_size = (params->batch_size == 0) ? dataset->n_samples : params->batch_size;
    if (batch_size > dataset->n_samples) {
        batch_size = dataset->n_samples;
    }

    nn_backprop_buffer *buf = NULL;
    nn_gradient *grad = NULL;
    size_t *order = NULL;
    nn_status status;

    status = nn_backprop_buffer_create(net, &buf);
    if (status != NN_OK) {
        return status;
    }
    status = nn_gradient_create(net, &grad);
    if (status != NN_OK) {
        nn_backprop_buffer_free(buf);
        return status;
    }
    order = malloc(dataset->n_samples * sizeof(*order));
    if (order == NULL) {
        nn_gradient_free(grad);
        nn_backprop_buffer_free(buf);
        return NN_ERR_ALLOC;
    }
    for (size_t s = 0; s < dataset->n_samples; s++) {
        order[s] = s;
    }

    nn_rng rng;
    nn_rng_seed(&rng, params->shuffle_seed);

    for (size_t epoch = 0; epoch < params->epochs; epoch++) {
        nn_shuffle_indices(order, dataset->n_samples, &rng);

        for (size_t batch_start = 0; batch_start < dataset->n_samples; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > dataset->n_samples) {
                batch_end = dataset->n_samples;
            }

            nn_gradient_zero(grad);
            for (size_t k = batch_start; k < batch_end; k++) {
                size_t s = order[k];
                const nn_real *sample_in = &dataset->inputs[s * dataset->n_inputs];
                const nn_real *sample_target = &dataset->targets[s * dataset->n_outputs];
                status = nn_backprop(net, buf, sample_in, sample_target, params->loss, grad);
                if (status != NN_OK) {
                    goto done;
                }
            }
            nn_gradient_apply(net, grad, params->learning_rate, batch_end - batch_start);
        }
    }
    status = NN_OK;

done:
    free(order);
    nn_gradient_free(grad);
    nn_backprop_buffer_free(buf);
    return status;
}
