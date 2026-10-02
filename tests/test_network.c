#include <stddef.h>

#include "nn.h"
#include "test.h"

static nn_status make_default_net(nn_network **out)
{
    size_t sizes[] = {2, 8, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_TANH, NN_ACT_SIGMOID};
    return nn_create(sizes, 4, acts, 3, NN_INIT_XAVIER, 1234, out);
}

static void test_create_architecture(void)
{
    nn_network *net = NULL;
    nn_status status = make_default_net(&net);
    NN_ASSERT(status == NN_OK);
    NN_ASSERT(net != NULL);
    NN_ASSERT(net->n_layers == 3);

    NN_ASSERT(net->layers[0].n_inputs == 2);
    NN_ASSERT(net->layers[0].n_outputs == 8);
    NN_ASSERT(net->layers[0].activation == NN_ACT_TANH);

    NN_ASSERT(net->layers[1].n_inputs == 8);
    NN_ASSERT(net->layers[1].n_outputs == 8);

    NN_ASSERT(net->layers[2].n_inputs == 8);
    NN_ASSERT(net->layers[2].n_outputs == 1);
    NN_ASSERT(net->layers[2].activation == NN_ACT_SIGMOID);

    nn_free(net);
}

static void test_create_rejects_bad_args(void)
{
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;

    NN_ASSERT(nn_create(NULL, 3, acts, 2, NN_INIT_UNIFORM, 0, &net) == NN_ERR_NULL_ARG);
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_UNIFORM, 0, NULL) == NN_ERR_NULL_ARG);

    NN_ASSERT(nn_create(sizes, 1, acts, 2, NN_INIT_UNIFORM, 0, &net) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_create(sizes, 3, acts, 1, NN_INIT_UNIFORM, 0, &net) == NN_ERR_INVALID_ARG);

    size_t bad_sizes[] = {2, 0, 1};
    NN_ASSERT(nn_create(bad_sizes, 3, acts, 2, NN_INIT_UNIFORM, 0, &net) == NN_ERR_INVALID_ARG);
}

static void test_weights_seeded_deterministically(void)
{
    nn_network *a = NULL;
    nn_network *b = NULL;
    size_t sizes[] = {2, 4, 1};
    nn_activation acts[] = {NN_ACT_RELU, NN_ACT_LINEAR};

    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_HE, 777, &a) == NN_OK);
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_HE, 777, &b) == NN_OK);

    for (size_t i = 0; i < a->layers[0].n_inputs * a->layers[0].n_outputs; i++) {
        NN_ASSERT_NEAR(a->layers[0].weights[i], b->layers[0].weights[i], 1e-15);
    }

    nn_free(a);
    nn_free(b);
}

static void test_biases_start_at_zero(void)
{
    nn_network *net = NULL;
    NN_ASSERT(make_default_net(&net) == NN_OK);

    for (size_t l = 0; l < net->n_layers; l++) {
        for (size_t b = 0; b < net->layers[l].n_outputs; b++) {
            NN_ASSERT(net->layers[l].biases[b] == 0.0);
        }
    }

    nn_free(net);
}

static void test_free_null_is_noop(void)
{
    nn_free(NULL);
    NN_ASSERT(1); /* reaching here without crashing is the test */
}

static void test_clone_is_deep_copy(void)
{
    nn_network *net = NULL;
    nn_network *copy = NULL;
    NN_ASSERT(make_default_net(&net) == NN_OK);
    NN_ASSERT(nn_clone(net, &copy) == NN_OK);

    NN_ASSERT(copy != net);
    NN_ASSERT(copy->n_layers == net->n_layers);

    for (size_t l = 0; l < net->n_layers; l++) {
        NN_ASSERT(copy->layers[l].weights != net->layers[l].weights);
        for (size_t w = 0; w < net->layers[l].n_inputs * net->layers[l].n_outputs; w++) {
            NN_ASSERT_NEAR(copy->layers[l].weights[w], net->layers[l].weights[w], 1e-15);
        }
    }

    /* Mutating the original must not affect the clone. */
    net->layers[0].weights[0] = 42.0;
    NN_ASSERT(copy->layers[0].weights[0] != 42.0);

    nn_free(net);
    nn_free(copy);
}

int main(void)
{
    NN_RUN(test_create_architecture);
    NN_RUN(test_create_rejects_bad_args);
    NN_RUN(test_weights_seeded_deterministically);
    NN_RUN(test_biases_start_at_zero);
    NN_RUN(test_free_null_is_noop);
    NN_RUN(test_clone_is_deep_copy);
    return NN_TEST_RESULT();
}
