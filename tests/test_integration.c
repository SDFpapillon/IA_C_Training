#include <math.h>

#include "nn.h"
#include "test.h"

#define INTEGRATION_PI 3.14159265358979323846

/*
 * XOR via genetic training. The supervised counterpart already lives in
 * tests/test_train.c (test_train_supervised_learns_xor) — this is the
 * "AND genetic" half of TODO.md's "learn XOR (supervised AND genetic)".
 */
typedef struct {
    nn_forward_buffer *buf;
    nn_real *scratch;
    const nn_dataset *ds;
} xor_fitness_ctx;

static nn_real xor_fitness(const nn_network *net, void *ctx_)
{
    xor_fitness_ctx *ctx = ctx_;
    nn_real mse = 0.0;
    nn_dataset_loss(net, ctx->buf, ctx->scratch, ctx->ds, NN_LOSS_MSE, &mse);
    return -mse; /* continuous signal: coarse accuracy as fitness causes premature
                  * convergence plateaus, as found while building Phase 5's demo */
}

static void test_xor_genetic_integration(void)
{
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(4, 2, 1, &ds) == NN_OK);
    nn_real inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    nn_real targets[] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; i++) ds->inputs[i] = inputs[i];
    for (size_t i = 0; i < 4; i++) ds->targets[i] = targets[i];

    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, 60, 1, &pop) == NN_OK);

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(pop->networks[0], &buf) == NN_OK);
    nn_real scratch[1];
    xor_fitness_ctx ctx = {buf, scratch, ds};

    nn_genetic_params params = {0};
    params.generations = 150;
    params.elitism = 2;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 4;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.15;
    params.mutation_stddev = 0.4;
    params.seed = 1;

    NN_ASSERT(nn_train_genetic(pop, xor_fitness, &ctx, &params) == NN_OK);

    size_t best = 0;
    for (size_t i = 1; i < pop->size; i++) {
        if (pop->fitness[i] > pop->fitness[best]) best = i;
    }
    nn_real acc = 0.0;
    NN_ASSERT(nn_dataset_accuracy(pop->networks[best], buf, scratch, ds, &acc) == NN_OK);
    NN_ASSERT(acc >= 0.99); /* XOR has 4 samples: accuracy lands on a multiple of 0.25 */

    nn_forward_buffer_free(buf);
    nn_population_free(pop);
    nn_dataset_free(ds);
}

/*
 * Two spirals: a classic hard benchmark for small MLPs. Kept deliberately
 * small (one turn, 30 points/class) so it trains in about a second and
 * stays reliable across seeds rather than chasing the full, tightly-wound
 * version real spiral demos use.
 */
#define SPIRAL_N_PER_CLASS 30
#define SPIRAL_TURNS 1.0

static void build_two_spirals(nn_dataset **out)
{
    NN_ASSERT(nn_dataset_create(SPIRAL_N_PER_CLASS * 2, 2, 1, out) == NN_OK);
    nn_dataset *ds = *out;
    for (size_t i = 0; i < SPIRAL_N_PER_CLASS; i++) {
        nn_real t = (nn_real)i / (nn_real)(SPIRAL_N_PER_CLASS - 1);
        nn_real angle = t * SPIRAL_TURNS * 2.0 * INTEGRATION_PI;
        nn_real radius = t;

        ds->inputs[i * 2 + 0] = radius * cos((double)angle);
        ds->inputs[i * 2 + 1] = radius * sin((double)angle);
        ds->targets[i] = 0.0;

        size_t j = SPIRAL_N_PER_CLASS + i;
        ds->inputs[j * 2 + 0] = radius * cos((double)angle + INTEGRATION_PI);
        ds->inputs[j * 2 + 1] = radius * sin((double)angle + INTEGRATION_PI);
        ds->targets[j] = 1.0;
    }
}

static void test_two_spirals_classification(void)
{
    nn_dataset *ds = NULL;
    build_two_spirals(&ds);

    size_t sizes[] = {2, 32, 32, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 4, acts, 3, NN_INIT_XAVIER, 1, &net) == NN_OK);

    nn_train_params params = {0};
    params.learning_rate = 0.3;
    params.epochs = 5000;
    params.batch_size = 0;
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 1;

    NN_ASSERT(nn_train_supervised(net, ds, &params) == NN_OK);

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(net, &buf) == NN_OK);
    nn_real scratch[1];
    nn_real acc = 0.0;
    NN_ASSERT(nn_dataset_accuracy(net, buf, scratch, ds, &acc) == NN_OK);
    /* Empirically 0.93-0.98 across seeds 1-5; 0.9 leaves a safe margin. */
    NN_ASSERT(acc >= 0.9);

    nn_forward_buffer_free(buf);
    nn_free(net);
    nn_dataset_free(ds);
}

int main(void)
{
    NN_RUN(test_xor_genetic_integration);
    NN_RUN(test_two_spirals_classification);
    return NN_TEST_RESULT();
}
