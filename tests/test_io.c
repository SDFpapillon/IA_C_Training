#include <stdio.h>

#include "nn.h"
#include "test.h"

#define NN_TEST_IO_PATH "build/nn_test_io_tmp.txt"

static nn_status make_test_net(nn_network **out)
{
    size_t sizes[] = {2, 3, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    return nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 55, out);
}

static void test_save_load_round_trip(void)
{
    nn_network *net = NULL;
    nn_network *loaded = NULL;

    NN_ASSERT(make_test_net(&net) == NN_OK);
    NN_ASSERT(nn_save(net, NN_TEST_IO_PATH) == NN_OK);
    NN_ASSERT(nn_load(NN_TEST_IO_PATH, &loaded) == NN_OK);

    NN_ASSERT(loaded->n_layers == net->n_layers);
    for (size_t l = 0; l < net->n_layers; l++) {
        NN_ASSERT(loaded->layers[l].n_inputs == net->layers[l].n_inputs);
        NN_ASSERT(loaded->layers[l].n_outputs == net->layers[l].n_outputs);
        NN_ASSERT(loaded->layers[l].activation == net->layers[l].activation);

        size_t n_weights = net->layers[l].n_inputs * net->layers[l].n_outputs;
        for (size_t w = 0; w < n_weights; w++) {
            NN_ASSERT_NEAR(loaded->layers[l].weights[w], net->layers[l].weights[w], 1e-12);
        }
        for (size_t b = 0; b < net->layers[l].n_outputs; b++) {
            NN_ASSERT_NEAR(loaded->layers[l].biases[b], net->layers[l].biases[b], 1e-12);
        }
    }

    nn_free(net);
    nn_free(loaded);
    remove(NN_TEST_IO_PATH);
}

static void test_load_missing_file_is_io_error(void)
{
    nn_network *net = NULL;
    NN_ASSERT(nn_load("build/does_not_exist.txt", &net) == NN_ERR_IO);
}

static void test_load_corrupt_file_is_format_error(void)
{
    const char *path = "build/nn_test_io_corrupt.txt";
    FILE *f = fopen(path, "w");
    NN_ASSERT(f != NULL);
    fprintf(f, "not a valid network file\n");
    fclose(f);

    nn_network *net = NULL;
    NN_ASSERT(nn_load(path, &net) == NN_ERR_FORMAT);

    remove(path);
}

int main(void)
{
    NN_RUN(test_save_load_round_trip);
    NN_RUN(test_load_missing_file_is_io_error);
    NN_RUN(test_load_corrupt_file_is_format_error);
    return NN_TEST_RESULT();
}
