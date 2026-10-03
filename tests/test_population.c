#include "nn.h"
#include "test.h"

static void test_create_random_basic(void)
{
    size_t sizes[] = {2, 4, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, 10, 42, &pop) == NN_OK);
    NN_ASSERT(pop->size == 10);
    NN_ASSERT(pop->genome_length == nn_genome_size(pop->networks[0]));

    /* Different individuals get different seeds, so (almost certainly) different weights. */
    NN_ASSERT(pop->networks[0]->layers[0].weights[0] != pop->networks[1]->layers[0].weights[0]);

    nn_population_free(pop);
}

static void test_create_random_rejects_bad_args(void)
{
    size_t sizes[] = {2, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(NULL, 2, acts, 1, NN_INIT_UNIFORM, 5, 1, &pop) == NN_ERR_NULL_ARG);
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 0, 1, &pop) == NN_ERR_INVALID_ARG);
}

static void test_create_seeded_individual_zero_is_exact_copy(void)
{
    size_t sizes[] = {2, 4, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *seed_net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 7, &seed_net) == NN_OK);

    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_seeded(seed_net, 6, 0.5, 1.0, 99, &pop) == NN_OK);

    size_t n_w = seed_net->layers[0].n_inputs * seed_net->layers[0].n_outputs;
    for (size_t w = 0; w < n_w; w++) {
        NN_ASSERT_NEAR(pop->networks[0]->layers[0].weights[w], seed_net->layers[0].weights[w], 1e-15);
    }

    /* At least one mutated individual should differ from the seed. */
    int any_different = 0;
    for (size_t i = 1; i < pop->size; i++) {
        for (size_t w = 0; w < n_w; w++) {
            if (pop->networks[i]->layers[0].weights[w] != seed_net->layers[0].weights[w]) {
                any_different = 1;
            }
        }
    }
    NN_ASSERT(any_different);

    nn_population_free(pop);
    nn_free(seed_net);
}

static void test_free_null_is_noop(void)
{
    nn_population_free(NULL);
    NN_ASSERT(1);
}

int main(void)
{
    NN_RUN(test_create_random_basic);
    NN_RUN(test_create_random_rejects_bad_args);
    NN_RUN(test_create_seeded_individual_zero_is_exact_copy);
    NN_RUN(test_free_null_is_noop);
    return NN_TEST_RESULT();
}
