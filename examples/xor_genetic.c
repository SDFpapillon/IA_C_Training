/*
 * The same XOR task as xor_supervised.c, solved by evolution instead of
 * gradient descent: a population of random 2-8-1 networks is bred toward
 * lower error over several generations. Fitness is -MSE (continuous)
 * rather than accuracy (0/0.25/0.5/0.75/1): a coarse, step-shaped fitness
 * gives genetic search far less to climb and tends to stall early.
 */
#include <stdio.h>

#include "nn.h"

typedef struct {
    nn_forward_buffer *buf;
    nn_real *scratch;
    const nn_dataset *ds;
} fitness_ctx;

static nn_real fitness(const nn_network *net, void *ctx_)
{
    fitness_ctx *ctx = ctx_;
    nn_real mse = 0.0;
    nn_dataset_loss(net, ctx->buf, ctx->scratch, ctx->ds, NN_LOSS_MSE, &mse);
    return -mse;
}

int main(void)
{
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};

    nn_dataset *ds = NULL;
    nn_dataset_create(4, 2, 1, &ds);
    nn_real inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    nn_real targets[] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; i++) ds->inputs[i] = inputs[i];
    for (size_t i = 0; i < 4; i++) ds->targets[i] = targets[i];

    nn_population *pop = NULL;
    nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, /*pop_size=*/60, /*seed=*/1, &pop);

    nn_forward_buffer *buf = NULL;
    nn_forward_buffer_create(pop->networks[0], &buf);
    nn_real scratch[1];
    fitness_ctx ctx = {buf, scratch, ds};

    nn_genetic_params params = {0};
    params.generations = 150;
    params.elitism = 2;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 4;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.15;
    params.mutation_stddev = 0.4;
    params.seed = 1;
    nn_train_genetic(pop, fitness, &ctx, &params);

    /* Find and run the best individual. */
    size_t best = 0;
    for (size_t i = 1; i < pop->size; i++) {
        if (pop->fitness[i] > pop->fitness[best]) best = i;
    }
    printf("best individual: #%zu, fitness (-MSE) = %.6f\n", best, (double)pop->fitness[best]);
    for (size_t i = 0; i < 4; i++) {
        nn_real output[1];
        nn_forward(pop->networks[best], buf, &ds->inputs[i * 2], output);
        printf("%g XOR %g = %.3f (expected %g)\n", (double)ds->inputs[i * 2],
               (double)ds->inputs[i * 2 + 1], (double)output[0], (double)ds->targets[i]);
    }

    nn_forward_buffer_free(buf);
    nn_population_free(pop);
    nn_dataset_free(ds);
    return 0;
}
