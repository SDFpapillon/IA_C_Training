/*
 * Compares pure genetic training against the three hybrid strategies from
 * TODO.md Phase 5, on a small 3-bit parity task (8 input combinations;
 * target is 1 if an odd number of bits are set). Two of those combinations
 * ({0,0,0} and {1,1,1}) are treated as "known cases" -- answers the
 * population shouldn't have to rediscover by chance, standing in for
 * TODO's "mate in 1 is always the best move" example.
 *
 * At each generation checkpoint, a fresh population is evolved from
 * scratch (same seed) for that many generations, and the best individual's
 * full-task accuracy is recorded. This is a comparison tool to run and
 * read, not a pass/fail check -- see tests/test_hybrid.c for the
 * deterministic correctness tests of each strategy's wiring.
 */
#include <stdio.h>

#include "nn.h"

#define POP_SIZE 40
#define HIDDEN 6
#define SEED 777

static const size_t CHECKPOINTS[] = {5, 10, 20, 30, 40, 60, 80};
#define N_CHECKPOINTS (sizeof(CHECKPOINTS) / sizeof(CHECKPOINTS[0]))

typedef struct {
    nn_forward_buffer *buf;
    nn_real *scratch;
    const nn_dataset *full;
    const nn_dataset *known;
    nn_real bonus; /* 0 to disable the Strategy 1 bonus */
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
    nn_dataset_create(8, 3, 1, &full);
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
    nn_dataset_create(2, 3, 1, &known);
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

static nn_real best_accuracy(nn_population *pop, nn_forward_buffer *buf, nn_real *scratch,
                              const nn_dataset *full)
{
    nn_real best = -1.0;
    for (size_t i = 0; i < pop->size; i++) {
        nn_real acc = 0.0;
        nn_dataset_accuracy(pop->networks[i], buf, scratch, full, &acc);
        if (acc > best) {
            best = acc;
        }
    }
    return best;
}

static void base_genetic_params(nn_genetic_params *params)
{
    *params = (nn_genetic_params){0};
    params->elitism = 2;
    params->selection = NN_SELECT_TOURNAMENT;
    params->tournament_size = 4;
    params->crossover = NN_CROSSOVER_UNIFORM;
    params->mutation_rate = 0.15;
    params->mutation_stddev = 0.4;
    params->seed = 123;
}

/* Pure genetic: random init, plain task accuracy as fitness. */
static nn_real run_pure(size_t generations, const size_t *sizes, const nn_activation *acts,
                         nn_forward_buffer *buf, nn_real *scratch, const nn_dataset *full)
{
    nn_population *pop = NULL;
    nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, POP_SIZE, SEED, &pop);

    fitness_ctx ctx = {buf, scratch, full, NULL, 0.0};
    nn_genetic_params params;
    base_genetic_params(&params);
    params.generations = generations;
    nn_train_genetic(pop, hybrid_fitness, &ctx, &params);

    nn_real acc = best_accuracy(pop, buf, scratch, full);
    nn_population_free(pop);
    return acc;
}

/* Strategy 1: fitness bonus for accuracy on the known cases. */
static nn_real run_strategy1(size_t generations, const size_t *sizes, const nn_activation *acts,
                              nn_forward_buffer *buf, nn_real *scratch, const nn_dataset *full,
                              const nn_dataset *known)
{
    nn_population *pop = NULL;
    nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, POP_SIZE, SEED, &pop);

    fitness_ctx ctx = {buf, scratch, full, known, 0.5};
    nn_genetic_params params;
    base_genetic_params(&params);
    params.generations = generations;
    nn_train_genetic(pop, hybrid_fitness, &ctx, &params);

    nn_real acc = best_accuracy(pop, buf, scratch, full);
    nn_population_free(pop);
    return acc;
}

/* Strategy 2: supervised pretraining on the known cases, then seed the population. */
static nn_real run_strategy2(size_t generations, const size_t *sizes, const nn_activation *acts,
                              nn_forward_buffer *buf, nn_real *scratch, const nn_dataset *full,
                              const nn_dataset *known)
{
    nn_network *pretrained = NULL;
    nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, SEED, &pretrained);
    nn_train_params sup_params = {0};
    sup_params.learning_rate = 0.5;
    sup_params.epochs = 300;
    sup_params.batch_size = 0;
    sup_params.loss = NN_LOSS_MSE;
    sup_params.shuffle_seed = 1;
    nn_train_supervised(pretrained, known, &sup_params);

    nn_population *pop = NULL;
    nn_population_create_seeded(pretrained, POP_SIZE, 0.15, 0.4, SEED, &pop);

    fitness_ctx ctx = {buf, scratch, full, NULL, 0.0};
    nn_genetic_params params;
    base_genetic_params(&params);
    params.generations = generations;
    nn_train_genetic(pop, hybrid_fitness, &ctx, &params);

    nn_real acc = best_accuracy(pop, buf, scratch, full);
    nn_population_free(pop);
    nn_free(pretrained);
    return acc;
}

/* Strategy 3: memetic -- a few gradient steps on the known cases per individual per generation. */
static nn_real run_strategy3(size_t generations, const size_t *sizes, const nn_activation *acts,
                              nn_forward_buffer *buf, nn_real *scratch, const nn_dataset *full,
                              const nn_dataset *known, nn_network *shape_net)
{
    nn_population *pop = NULL;
    nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, POP_SIZE, SEED, &pop);

    memetic_ctx mctx;
    nn_backprop_buffer_create(shape_net, &mctx.buf);
    nn_gradient_create(shape_net, &mctx.grad);
    mctx.known = known;
    mctx.learning_rate = 0.5;
    mctx.steps = 3;

    fitness_ctx ctx = {buf, scratch, full, NULL, 0.0};
    nn_genetic_params params;
    base_genetic_params(&params);
    params.generations = generations;
    params.pre_eval_hook = memetic_hook;
    params.pre_eval_ctx = &mctx;
    nn_train_genetic(pop, hybrid_fitness, &ctx, &params);

    nn_real acc = best_accuracy(pop, buf, scratch, full);
    nn_backprop_buffer_free(mctx.buf);
    nn_gradient_free(mctx.grad);
    nn_population_free(pop);
    return acc;
}

int main(void)
{
    size_t sizes[] = {3, HIDDEN, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};

    nn_dataset *full = NULL;
    nn_dataset *known = NULL;
    build_parity_datasets(&full, &known);

    nn_network *shape_net = NULL;
    nn_create(sizes, 3, acts, 2, NN_INIT_UNIFORM, 0, &shape_net);
    nn_forward_buffer *buf = NULL;
    nn_forward_buffer_create(shape_net, &buf);
    nn_real scratch[1];

    printf("3-bit parity, best full-task accuracy vs. generation budget\n");
    printf("%-12s%-10s%-10s%-10s%-10s\n", "generations", "pure", "strat1", "strat2", "strat3");
    for (size_t c = 0; c < N_CHECKPOINTS; c++) {
        size_t gens = CHECKPOINTS[c];
        nn_real acc_pure = run_pure(gens, sizes, acts, buf, scratch, full);
        nn_real acc_s1 = run_strategy1(gens, sizes, acts, buf, scratch, full, known);
        nn_real acc_s2 = run_strategy2(gens, sizes, acts, buf, scratch, full, known);
        nn_real acc_s3 = run_strategy3(gens, sizes, acts, buf, scratch, full, known, shape_net);
        printf("%-12zu%-10.3f%-10.3f%-10.3f%-10.3f\n", gens, (double)acc_pure, (double)acc_s1,
               (double)acc_s2, (double)acc_s3);
    }

    nn_forward_buffer_free(buf);
    nn_free(shape_net);
    nn_dataset_free(full);
    nn_dataset_free(known);
    return 0;
}
