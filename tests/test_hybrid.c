#include <stdlib.h>

#include "nn.h"
#include "test.h"

/*
 * These tests check that the three hybrid strategies from TODO.md Phase 5
 * are wired correctly (bonus math, seeding, and the gradient hook actually
 * improve the network they touch). They deliberately do NOT assert that a
 * full genetic run with a given strategy "beats" pure genetic on some toy
 * task: GA outcomes on a fixed, tiny generation/population budget are
 * sensitive to the task and hyperparameters, so hard-coding a specific race
 * result would just be fitting the test to one lucky configuration rather
 * than validating real behavior. Running and comparing the strategies
 * end-to-end is instead a job for examples/hybrid_training_demo.c.
 */

/* Strategy 1: task score (negative MSE, continuous) + bonus * accuracy on
 * known cases. */
typedef struct {
    nn_forward_buffer *buf;
    nn_real *scratch;
    const nn_dataset *full;
    const nn_dataset *known;
    nn_real bonus;
} fitness_ctx;

static nn_real hybrid_fitness(const nn_network *net, void *ctx_)
{
    fitness_ctx *ctx = ctx_;
    nn_real mse = 0.0;
    nn_dataset_loss(net, ctx->buf, ctx->scratch, ctx->full, NN_LOSS_MSE, &mse);
    nn_real score = -mse;
    if (ctx->known != NULL && ctx->bonus != 0.0) {
        nn_real known_acc = 0.0;
        nn_dataset_accuracy(net, ctx->buf, ctx->scratch, ctx->known, &known_acc);
        score += ctx->bonus * known_acc;
    }
    return score;
}

/* Strategy 3: a few gradient steps on the known cases, run on each
 * individual right before its fitness is evaluated. */
typedef struct {
    nn_backprop_buffer *buf;
    nn_gradient *grad;
    const nn_dataset *known;
    nn_real learning_rate;
    size_t steps;
} memetic_ctx;

static void memetic_hook(nn_network *net, void *ctx_)
{
    memetic_ctx *ctx = ctx_;
    for (size_t step = 0; step < ctx->steps; step++) {
        nn_gradient_zero(ctx->grad);
        for (size_t s = 0; s < ctx->known->n_samples; s++) {
            const nn_real *in = &ctx->known->inputs[s * ctx->known->n_inputs];
            const nn_real *target = &ctx->known->targets[s * ctx->known->n_outputs];
            nn_backprop(net, ctx->buf, in, target, NN_LOSS_MSE, ctx->grad);
        }
        nn_gradient_apply(net, ctx->grad, ctx->learning_rate, ctx->known->n_samples);
    }
}

static void build_parity_datasets(nn_dataset **full_out, nn_dataset **known_out)
{
    nn_dataset *full = NULL;
    NN_ASSERT(nn_dataset_create(8, 3, 1, &full) == NN_OK);
    for (size_t i = 0; i < 8; i++) {
        int b0 = (int)((i >> 2) & 1u);
        int b1 = (int)((i >> 1) & 1u);
        int b2 = (int)(i & 1u);
        full->inputs[i * 3 + 0] = (nn_real)b0;
        full->inputs[i * 3 + 1] = (nn_real)b1;
        full->inputs[i * 3 + 2] = (nn_real)b2;
        full->targets[i] = (nn_real)(b0 ^ b1 ^ b2);
    }

    nn_dataset *known = NULL;
    NN_ASSERT(nn_dataset_create(2, 3, 1, &known) == NN_OK);
    known->inputs[0] = 0.0;
    known->inputs[1] = 0.0;
    known->inputs[2] = 0.0;
    known->targets[0] = 0.0;
    known->inputs[3] = 1.0;
    known->inputs[4] = 1.0;
    known->inputs[5] = 1.0;
    known->targets[1] = 1.0;

    *full_out = full;
    *known_out = known;
}

static void test_strategy1_fitness_bonus_is_additive(void)
{
    /* Weights and bias zero: output is always 0. */
    nn_real weights[] = {0.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *full = NULL;
    NN_ASSERT(nn_dataset_create(2, 1, 1, &full) == NN_OK);
    full->inputs[0] = 0.0;
    full->targets[0] = 0.0; /* (0-0)^2 = 0 */
    full->inputs[1] = 0.0;
    full->targets[1] = 1.0; /* (0-1)^2 = 1; MSE = (0+1)/2 = 0.5 */

    nn_dataset *known = NULL;
    NN_ASSERT(nn_dataset_create(1, 1, 1, &known) == NN_OK);
    known->inputs[0] = 0.0;
    known->targets[0] = 0.0; /* output 0 matches (both <= 0.5): accuracy 1.0 */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real scratch[1];

    fitness_ctx ctx = {buf, scratch, full, known, 0.5};
    nn_real fitness = hybrid_fitness(&net, &ctx);

    /* -MSE(full) + bonus * accuracy(known) = -0.5 + 0.5 * 1.0 = 0.0 */
    NN_ASSERT_NEAR(fitness, 0.0, 1e-12);

    nn_forward_buffer_free(buf);
    nn_dataset_free(full);
    nn_dataset_free(known);
}

static void test_strategy2_seeds_population_from_pretrained_network(void)
{
    size_t sizes[] = {3, 4, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_dataset *full = NULL;
    nn_dataset *known = NULL;
    build_parity_datasets(&full, &known);

    nn_network *pretrained = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &pretrained) == NN_OK);

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(pretrained, &buf) == NN_OK);
    nn_real scratch[1];
    nn_real loss_before = 0.0;
    NN_ASSERT(nn_dataset_loss(pretrained, buf, scratch, known, NN_LOSS_MSE, &loss_before) == NN_OK);

    nn_train_params sup_params = {0};
    sup_params.learning_rate = 0.5;
    sup_params.epochs = 300;
    sup_params.batch_size = 0;
    sup_params.loss = NN_LOSS_MSE;
    sup_params.shuffle_seed = 1;
    NN_ASSERT(nn_train_supervised(pretrained, known, &sup_params) == NN_OK);

    nn_real loss_after = 0.0;
    NN_ASSERT(nn_dataset_loss(pretrained, buf, scratch, known, NN_LOSS_MSE, &loss_after) == NN_OK);
    NN_ASSERT(loss_after < loss_before);

    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_seeded(pretrained, 5, 0.3, 0.5, 42, &pop) == NN_OK);

    /* Individual 0 must be an exact, unmutated copy of the pretrained network. */
    size_t n = nn_genome_size(pretrained);
    nn_real *g_seed = malloc(n * sizeof(*g_seed));
    nn_real *g_indiv0 = malloc(n * sizeof(*g_indiv0));
    NN_ASSERT(g_seed != NULL && g_indiv0 != NULL);
    nn_genome_flatten(pretrained, g_seed);
    nn_genome_flatten(pop->networks[0], g_indiv0);
    for (size_t i = 0; i < n; i++) {
        NN_ASSERT_NEAR(g_seed[i], g_indiv0[i], 1e-15);
    }

    free(g_seed);
    free(g_indiv0);
    nn_population_free(pop);
    nn_forward_buffer_free(buf);
    nn_free(pretrained);
    nn_dataset_free(full);
    nn_dataset_free(known);
}

static void test_strategy3_memetic_hook_reduces_known_case_loss(void)
{
    size_t sizes[] = {3, 4, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_dataset *full = NULL;
    nn_dataset *known = NULL;
    build_parity_datasets(&full, &known);

    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 7, &net) == NN_OK);

    nn_forward_buffer *fbuf = NULL;
    NN_ASSERT(nn_forward_buffer_create(net, &fbuf) == NN_OK);
    nn_real scratch[1];
    nn_real loss_before = 0.0;
    NN_ASSERT(nn_dataset_loss(net, fbuf, scratch, known, NN_LOSS_MSE, &loss_before) == NN_OK);

    memetic_ctx mctx;
    NN_ASSERT(nn_backprop_buffer_create(net, &mctx.buf) == NN_OK);
    NN_ASSERT(nn_gradient_create(net, &mctx.grad) == NN_OK);
    mctx.known = known;
    mctx.learning_rate = 0.5;
    mctx.steps = 5;

    memetic_hook(net, &mctx);

    nn_real loss_after = 0.0;
    NN_ASSERT(nn_dataset_loss(net, fbuf, scratch, known, NN_LOSS_MSE, &loss_after) == NN_OK);
    NN_ASSERT(loss_after < loss_before);

    nn_backprop_buffer_free(mctx.buf);
    nn_gradient_free(mctx.grad);
    nn_forward_buffer_free(fbuf);
    nn_free(net);
    nn_dataset_free(full);
    nn_dataset_free(known);
}

static size_t g_hook_call_count = 0;

static void counting_hook(nn_network *net, void *ctx)
{
    (void)net;
    (void)ctx;
    g_hook_call_count++;
}

static nn_real zero_fitness(const nn_network *net, void *ctx)
{
    (void)net;
    (void)ctx;
    return 0.0;
}

static void test_pre_eval_hook_called_once_per_individual_per_generation(void)
{
    g_hook_call_count = 0;

    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 5, 1, &pop) == NN_OK);

    nn_genetic_params params = {0};
    params.generations = 4;
    params.elitism = 1;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 2;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.1;
    params.mutation_stddev = 0.1;
    params.seed = 1;
    params.pre_eval_hook = counting_hook;

    NN_ASSERT(nn_train_genetic(pop, zero_fitness, NULL, &params) == NN_OK);
    NN_ASSERT(g_hook_call_count == pop->size * params.generations);

    nn_population_free(pop);
}

static void test_strategy1_runs_end_to_end(void)
{
    size_t sizes[] = {3, 4, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_dataset *full = NULL;
    nn_dataset *known = NULL;
    build_parity_datasets(&full, &known);

    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, 10, 1, &pop) == NN_OK);
    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(pop->networks[0], &buf) == NN_OK);
    nn_real scratch[1];
    fitness_ctx ctx = {buf, scratch, full, known, 0.5};

    nn_genetic_params params = {0};
    params.generations = 5;
    params.elitism = 1;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 3;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.1;
    params.mutation_stddev = 0.3;
    params.seed = 5;

    NN_ASSERT(nn_train_genetic(pop, hybrid_fitness, &ctx, &params) == NN_OK);
    for (size_t i = 0; i < pop->size; i++) {
        NN_ASSERT(pop->fitness[i] > -1e9); /* sanity: populated with a real value */
    }

    nn_forward_buffer_free(buf);
    nn_population_free(pop);
    nn_dataset_free(full);
    nn_dataset_free(known);
}

int main(void)
{
    NN_RUN(test_strategy1_fitness_bonus_is_additive);
    NN_RUN(test_strategy2_seeds_population_from_pretrained_network);
    NN_RUN(test_strategy3_memetic_hook_reduces_known_case_loss);
    NN_RUN(test_pre_eval_hook_called_once_per_individual_per_generation);
    NN_RUN(test_strategy1_runs_end_to_end);
    return NN_TEST_RESULT();
}
