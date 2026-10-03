#include <stdio.h>
#include <string.h>

#include "nn.h"
#include "test.h"

#define NN_TEST_HISTORY_PATH "build/nn_test_history_tmp.csv"
#define NN_TEST_FITNESS_PATH "build/nn_test_fitness_tmp.csv"

static void test_history_logger_logs_expected_rows(void)
{
    /* Weights and bias zero: output is always 0. */
    nn_real weights[] = {0.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(1, 1, 1, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->targets[0] = 1.0; /* output 0 vs target 1: MSE = 1.0, accuracy = 0.0 */

    nn_history_logger *logger = NULL;
    NN_ASSERT(nn_history_logger_create(NN_TEST_HISTORY_PATH, &net, ds, NN_LOSS_MSE, &logger) == NN_OK);

    nn_history_log_step(&net, 0, logger);
    nn_history_log_step(&net, 5, logger);
    nn_history_logger_free(logger);

    FILE *f = fopen(NN_TEST_HISTORY_PATH, "r");
    NN_ASSERT(f != NULL);

    char header[64];
    NN_ASSERT(fgets(header, sizeof(header), f) != NULL);
    NN_ASSERT(strncmp(header, "iteration,loss,accuracy", strlen("iteration,loss,accuracy")) == 0);

    size_t iter;
    double loss, acc;
    NN_ASSERT(fscanf(f, "%zu,%lf,%lf\n", &iter, &loss, &acc) == 3);
    NN_ASSERT(iter == 0);
    NN_ASSERT_NEAR(loss, 1.0, 1e-12);
    NN_ASSERT_NEAR(acc, 0.0, 1e-12);

    NN_ASSERT(fscanf(f, "%zu,%lf,%lf\n", &iter, &loss, &acc) == 3);
    NN_ASSERT(iter == 5);
    NN_ASSERT_NEAR(loss, 1.0, 1e-12);
    NN_ASSERT_NEAR(acc, 0.0, 1e-12);

    fclose(f);
    remove(NN_TEST_HISTORY_PATH);
    nn_dataset_free(ds);
}

static void test_history_logger_create_rejects_null(void)
{
    nn_history_logger *logger = NULL;
    NN_ASSERT(nn_history_logger_create(NULL, NULL, NULL, NN_LOSS_MSE, &logger) == NN_ERR_NULL_ARG);
}

static void test_history_logger_free_null_is_noop(void)
{
    nn_history_logger_free(NULL);
    NN_ASSERT(1);
}

static void test_fitness_logger_logs_best_and_avg(void)
{
    size_t sizes[] = {1, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_population *pop = NULL;
    NN_ASSERT(nn_population_create_random(sizes, 2, acts, 1, NN_INIT_UNIFORM, 3, 1, &pop) == NN_OK);
    pop->fitness[0] = 1.0;
    pop->fitness[1] = 2.0;
    pop->fitness[2] = 6.0; /* best = 6, avg = (1+2+6)/3 = 3 */

    nn_fitness_logger *logger = NULL;
    NN_ASSERT(nn_fitness_logger_create(NN_TEST_FITNESS_PATH, &logger) == NN_OK);
    nn_fitness_log_generation(pop, 7, logger);
    nn_fitness_logger_free(logger);

    FILE *f = fopen(NN_TEST_FITNESS_PATH, "r");
    NN_ASSERT(f != NULL);

    char header[64];
    NN_ASSERT(fgets(header, sizeof(header), f) != NULL);
    NN_ASSERT(strncmp(header, "generation,best_fitness,avg_fitness",
                       strlen("generation,best_fitness,avg_fitness")) == 0);

    size_t gen;
    double best, avg;
    NN_ASSERT(fscanf(f, "%zu,%lf,%lf\n", &gen, &best, &avg) == 3);
    NN_ASSERT(gen == 7);
    NN_ASSERT_NEAR(best, 6.0, 1e-12);
    NN_ASSERT_NEAR(avg, 3.0, 1e-12);

    fclose(f);
    remove(NN_TEST_FITNESS_PATH);
    nn_population_free(pop);
}

static void test_fitness_logger_create_rejects_null(void)
{
    nn_fitness_logger *logger = NULL;
    NN_ASSERT(nn_fitness_logger_create(NULL, &logger) == NN_ERR_NULL_ARG);
}

static void test_fitness_logger_free_null_is_noop(void)
{
    nn_fitness_logger_free(NULL);
    NN_ASSERT(1);
}

int main(void)
{
    NN_RUN(test_history_logger_logs_expected_rows);
    NN_RUN(test_history_logger_create_rejects_null);
    NN_RUN(test_history_logger_free_null_is_noop);
    NN_RUN(test_fitness_logger_logs_best_and_avg);
    NN_RUN(test_fitness_logger_create_rejects_null);
    NN_RUN(test_fitness_logger_free_null_is_noop);
    return NN_TEST_RESULT();
}
