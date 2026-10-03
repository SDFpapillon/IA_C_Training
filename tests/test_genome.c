#include <stdlib.h>

#include "nn.h"
#include "test.h"

static void test_genome_size_matches_param_count(void)
{
    size_t sizes[] = {2, 3, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &net) == NN_OK);

    /* layer0: 2*3 weights + 3 biases = 9; layer1: 3*1 weights + 1 bias = 4; total 13 */
    NN_ASSERT(nn_genome_size(net) == 13);

    nn_free(net);
}

static void test_flatten_unflatten_round_trip(void)
{
    size_t sizes[] = {2, 3, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &net) == NN_OK);

    size_t n = nn_genome_size(net);
    nn_real *genome = malloc(n * sizeof(*genome));
    NN_ASSERT(genome != NULL);
    nn_genome_flatten(net, genome);

    /* Mutate the network directly, then restore it from the saved genome. */
    net->layers[0].weights[0] = 999.0;
    net->layers[1].biases[0] = -999.0;
    nn_genome_unflatten(net, genome);

    nn_real *genome2 = malloc(n * sizeof(*genome2));
    NN_ASSERT(genome2 != NULL);
    nn_genome_flatten(net, genome2);
    for (size_t i = 0; i < n; i++) {
        NN_ASSERT_NEAR(genome[i], genome2[i], 1e-15);
    }

    free(genome);
    free(genome2);
    nn_free(net);
}

static void test_flatten_order_is_weights_then_biases_per_layer(void)
{
    nn_real weights[] = {1.0, 2.0, 3.0, 4.0}; /* 2 outputs x 2 inputs */
    nn_real biases[] = {5.0, 6.0};
    nn_layer layer = {2, 2, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_real genome[6];
    nn_genome_flatten(&net, genome);
    nn_real expected[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (size_t i = 0; i < 6; i++) {
        NN_ASSERT_NEAR(genome[i], expected[i], 1e-15);
    }
}

int main(void)
{
    NN_RUN(test_genome_size_matches_param_count);
    NN_RUN(test_flatten_unflatten_round_trip);
    NN_RUN(test_flatten_order_is_weights_then_biases_per_layer);
    return NN_TEST_RESULT();
}
