#include <string.h>

#include "nn.h"
#include "test.h"

static void test_crossover_uniform_picks_from_either_parent(void)
{
    nn_real a[8], b[8], child[8];
    for (size_t i = 0; i < 8; i++) {
        a[i] = (nn_real)i;
        b[i] = (nn_real)(100 + i);
    }

    nn_rng rng;
    nn_rng_seed(&rng, 5);
    nn_crossover_apply(a, b, child, 8, NN_CROSSOVER_UNIFORM, &rng);

    for (size_t i = 0; i < 8; i++) {
        NN_ASSERT(child[i] == a[i] || child[i] == b[i]);
    }
}

static void test_crossover_one_point_has_single_switch(void)
{
    nn_real a[10], b[10], child[10];
    for (size_t i = 0; i < 10; i++) {
        a[i] = (nn_real)i;
        b[i] = (nn_real)(100 + i);
    }

    nn_rng rng;
    nn_rng_seed(&rng, 9);
    nn_crossover_apply(a, b, child, 10, NN_CROSSOVER_ONE_POINT, &rng);

    int switches = 0;
    for (size_t i = 1; i < 10; i++) {
        int prev_from_a = (child[i - 1] == a[i - 1]);
        int cur_from_a = (child[i] == a[i]);
        if (prev_from_a != cur_from_a) {
            switches++;
        }
    }
    NN_ASSERT(switches <= 1);
}

static void test_mutate_rate_zero_never_changes(void)
{
    nn_real genome[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    nn_real original[5];
    memcpy(original, genome, sizeof(genome));

    nn_rng rng;
    nn_rng_seed(&rng, 1);
    nn_mutate(genome, 5, 0.0, 10.0, &rng);

    for (size_t i = 0; i < 5; i++) {
        NN_ASSERT_NEAR(genome[i], original[i], 1e-15);
    }
}

static void test_mutate_rate_one_always_changes(void)
{
    nn_real genome[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    nn_real original[5];
    memcpy(original, genome, sizeof(genome));

    nn_rng rng;
    nn_rng_seed(&rng, 1);
    nn_mutate(genome, 5, 1.0, 1.0, &rng);

    int any_changed = 0;
    for (size_t i = 0; i < 5; i++) {
        if (genome[i] != original[i]) {
            any_changed = 1;
        }
    }
    NN_ASSERT(any_changed);
}

static void test_select_tournament_favors_higher_fitness(void)
{
    nn_real fitness[] = {0.0, 1.0, 2.0, 3.0, 100.0}; /* index 4 dominates */
    nn_rng rng;
    nn_rng_seed(&rng, 3);

    const int trials = 500;
    int best_count = 0;
    for (int t = 0; t < trials; t++) {
        size_t idx = nn_select(fitness, 5, NN_SELECT_TOURNAMENT, 3, 0.0, &rng);
        if (idx == 4) {
            best_count++;
        }
    }
    /* Random chance alone would give ~20% (100/500); tournament selection
     * should clear that comfortably even allowing for sampling noise. */
    NN_ASSERT(best_count > trials * 2 / 5);
}

static void test_select_roulette_favors_higher_fitness(void)
{
    nn_real fitness[] = {1.0, 1.0, 1.0, 1.0, 96.0};
    nn_real total = 100.0;
    nn_rng rng;
    nn_rng_seed(&rng, 4);

    const int trials = 500;
    int best_count = 0;
    for (int t = 0; t < trials; t++) {
        size_t idx = nn_select(fitness, 5, NN_SELECT_ROULETTE, 0, total, &rng);
        if (idx == 4) {
            best_count++;
        }
    }
    NN_ASSERT(best_count > trials / 2);
}

static void test_select_roulette_nonpositive_total_falls_back_to_uniform(void)
{
    nn_real fitness[] = {-1.0, -1.0, -1.0};
    nn_rng rng;
    nn_rng_seed(&rng, 2);

    for (int t = 0; t < 20; t++) {
        size_t idx = nn_select(fitness, 3, NN_SELECT_ROULETTE, 0, -3.0, &rng);
        NN_ASSERT(idx < 3);
    }
}

static nn_real dummy_fitness(const nn_network *net, void *ctx)
{
    (void)net;
    (void)ctx;
    return 0.0;
}

static void test_train_genetic_rejects_elitism_too_large(void)
{
    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 3, 1, &pop) == NN_OK);

    nn_genetic_params params = {0};
    params.generations = 1;
    params.elitism = 4; /* > pop size (3) */

    NN_ASSERT(nn_train_genetic(pop, dummy_fitness, NULL, &params) == NN_ERR_INVALID_ARG);

    nn_population_free(pop);
}

static nn_real neg_first_weight_fitness(const nn_network *net, void *ctx)
{
    (void)ctx;
    return -net->layers[0].weights[0];
}

static void test_elitism_preserves_the_best_genome_exactly(void)
{
    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 5, 123, &pop) == NN_OK);

    size_t genome_len = pop->genome_length;
    nn_real best_fitness = neg_first_weight_fitness(pop->networks[0], NULL);
    size_t best_idx = 0;
    for (size_t i = 1; i < pop->size; i++) {
        nn_real f = neg_first_weight_fitness(pop->networks[i], NULL);
        if (f > best_fitness) {
            best_fitness = f;
            best_idx = i;
        }
    }
    nn_real best_genome[8];
    nn_genome_flatten(pop->networks[best_idx], best_genome);

    nn_genetic_params params = {0};
    params.generations = 2; /* gen 0 breeds, gen 1 evaluates the bred population and stops */
    params.elitism = 1;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 2;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.0;
    params.mutation_stddev = 0.0;
    params.seed = 1;

    NN_ASSERT(nn_train_genetic(pop, neg_first_weight_fitness, NULL, &params) == NN_OK);

    nn_real final_genome[8];
    nn_genome_flatten(pop->networks[0], final_genome);
    for (size_t i = 0; i < genome_len; i++) {
        NN_ASSERT_NEAR(final_genome[i], best_genome[i], 1e-15);
    }

    nn_population_free(pop);
}

typedef struct {
    nn_forward_buffer *buf;
} evolve_ctx;

static nn_real target_output_fitness(const nn_network *net, void *ctx_)
{
    evolve_ctx *ctx = ctx_;
    nn_real input[] = {0.5, -0.5};
    nn_real target[] = {0.2, 0.8};
    nn_real output[2];
    nn_forward(net, ctx->buf, input, output);
    return -nn_loss_value(NN_LOSS_MSE, output, target, 2);
}

static void test_train_genetic_evolves_toward_target_output(void)
{
    size_t sizes[] = {2, 6, 2};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, 60, 7, &pop) == NN_OK);

    evolve_ctx ctx;
    NN_ASSERT(nn_forward_buffer_create(pop->networks[0], &ctx.buf) == NN_OK);

    nn_real initial_best = target_output_fitness(pop->networks[0], &ctx);
    for (size_t i = 1; i < pop->size; i++) {
        nn_real f = target_output_fitness(pop->networks[i], &ctx);
        if (f > initial_best) {
            initial_best = f;
        }
    }

    nn_genetic_params params = {0};
    params.generations = 300;
    params.elitism = 2;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 4;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.1;
    params.mutation_stddev = 0.3;
    params.seed = 42;

    NN_ASSERT(nn_train_genetic(pop, target_output_fitness, &ctx, &params) == NN_OK);

    nn_real final_best = pop->fitness[0];
    for (size_t i = 1; i < pop->size; i++) {
        if (pop->fitness[i] > final_best) {
            final_best = pop->fitness[i];
        }
    }

    NN_ASSERT(final_best > initial_best);
    NN_ASSERT(final_best > -0.01); /* MSE below 0.01: close to the target */

    nn_forward_buffer_free(ctx.buf);
    nn_population_free(pop);
}

int main(void)
{
    NN_RUN(test_crossover_uniform_picks_from_either_parent);
    NN_RUN(test_crossover_one_point_has_single_switch);
    NN_RUN(test_mutate_rate_zero_never_changes);
    NN_RUN(test_mutate_rate_one_always_changes);
    NN_RUN(test_select_tournament_favors_higher_fitness);
    NN_RUN(test_select_roulette_favors_higher_fitness);
    NN_RUN(test_select_roulette_nonpositive_total_falls_back_to_uniform);
    NN_RUN(test_train_genetic_rejects_elitism_too_large);
    NN_RUN(test_elitism_preserves_the_best_genome_exactly);
    NN_RUN(test_train_genetic_evolves_toward_target_output);
    return NN_TEST_RESULT();
}
