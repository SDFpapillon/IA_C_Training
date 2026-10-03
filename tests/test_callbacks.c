#include "nn.h"
#include "test.h"

static size_t g_step_calls = 0;
static size_t g_last_iteration = 0;

static void counting_step(const nn_network *net, size_t iteration, void *ctx)
{
    (void)net;
    (void)ctx;
    g_step_calls++;
    g_last_iteration = iteration;
}

static void test_on_step_fires_every_stride_during_supervised_training(void)
{
    g_step_calls = 0;
    g_last_iteration = 0;

    size_t sizes[] = {2, 2, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &net) == NN_OK);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(1, 2, 1, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->inputs[1] = 0.0;
    ds->targets[0] = 0.0;

    nn_train_params params = {0};
    params.learning_rate = 0.1;
    params.epochs = 10;
    params.batch_size = 0;
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 1;
    params.on_step = counting_step;
    params.on_step_every = 3;

    NN_ASSERT(nn_train_supervised(net, ds, &params) == NN_OK);
    /* epochs 0..9, every 3rd: 0, 3, 6, 9 -> 4 calls, last at iteration 9 */
    NN_ASSERT(g_step_calls == 4);
    NN_ASSERT(g_last_iteration == 9);

    nn_dataset_free(ds);
    nn_free(net);
}

static void test_on_step_not_called_when_null(void)
{
    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 2, acts, 1, NN_INIT_UNIFORM, 1, &net) == NN_OK);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(1, 1, 1, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->targets[0] = 0.0;

    nn_train_params params = {0};
    params.learning_rate = 0.1;
    params.epochs = 5;
    params.loss = NN_LOSS_MSE;
    /* on_step left NULL */

    NN_ASSERT(nn_train_supervised(net, ds, &params) == NN_OK);

    nn_dataset_free(ds);
    nn_free(net);
}

static size_t g_gen_calls = 0;
static size_t g_last_generation = 0;

static void counting_generation(const nn_population *pop, size_t generation, void *ctx)
{
    (void)pop;
    (void)ctx;
    g_gen_calls++;
    g_last_generation = generation;
}

static nn_real zero_fitness(const nn_network *net, void *ctx)
{
    (void)net;
    (void)ctx;
    return 0.0;
}

static void test_on_generation_fires_every_stride_during_genetic_training(void)
{
    g_gen_calls = 0;
    g_last_generation = 0;

    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 4, 1, &pop) == NN_OK);

    nn_genetic_params params = {0};
    params.generations = 5;
    params.elitism = 1;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 2;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.1;
    params.mutation_stddev = 0.1;
    params.seed = 1;
    params.on_generation = counting_generation;
    params.on_generation_every = 2;

    NN_ASSERT(nn_train_genetic(pop, zero_fitness, NULL, &params) == NN_OK);
    /* generations 0..4, every 2nd: 0, 2, 4 -> 3 calls, last at generation 4 */
    NN_ASSERT(g_gen_calls == 3);
    NN_ASSERT(g_last_generation == 4);

    nn_population_free(pop);
}

int main(void)
{
    NN_RUN(test_on_step_fires_every_stride_during_supervised_training);
    NN_RUN(test_on_step_not_called_when_null);
    NN_RUN(test_on_generation_fires_every_stride_during_genetic_training);
    return NN_TEST_RESULT();
}
