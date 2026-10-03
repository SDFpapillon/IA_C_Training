/*
 * The smallest possible supervised-training example: a 2-8-1 network
 * learns XOR by gradient descent. Start here if you're new to the API;
 * see visualize_training.c and spirals.c for the export/visualization
 * pieces, and xor_genetic.c for the same task solved by evolution instead.
 */
#include <stdio.h>

#include "nn.h"

int main(void)
{
    /* 2 inputs -> 8 hidden (tanh) -> 1 output (sigmoid). */
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, /*seed=*/1, &net);

    /* The 4 XOR cases. */
    nn_dataset *ds = NULL;
    nn_dataset_create(4, 2, 1, &ds);
    nn_real inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    nn_real targets[] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; i++) ds->inputs[i] = inputs[i];
    for (size_t i = 0; i < 4; i++) ds->targets[i] = targets[i];

    nn_train_params params = {0};
    params.learning_rate = 0.5;
    params.epochs = 2000;
    params.batch_size = 0; /* full-batch gradient descent */
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 1;
    nn_train_supervised(net, ds, &params);

    /* Run the trained network on each case and print the result. */
    nn_forward_buffer *buf = NULL;
    nn_forward_buffer_create(net, &buf);
    for (size_t i = 0; i < 4; i++) {
        nn_real output[1];
        nn_forward(net, buf, &ds->inputs[i * 2], output);
        printf("%g XOR %g = %.3f (expected %g)\n", (double)ds->inputs[i * 2],
               (double)ds->inputs[i * 2 + 1], (double)output[0], (double)ds->targets[i]);
    }

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
    nn_free(net);
    return 0;
}
