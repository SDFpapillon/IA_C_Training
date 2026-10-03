#include <stdio.h>

#include "nn.h"
#include "test.h"

#define NN_TEST_CSV_PATH "build/nn_test_dataset_tmp.csv"

static void test_create_and_free(void)
{
    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(3, 2, 1, &ds) == NN_OK);
    NN_ASSERT(ds->n_samples == 3);
    NN_ASSERT(ds->n_inputs == 2);
    NN_ASSERT(ds->n_outputs == 1);
    nn_dataset_free(ds);
}

static void test_create_rejects_bad_args(void)
{
    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(0, 2, 1, &ds) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_dataset_create(3, 0, 1, &ds) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_dataset_create(3, 2, 0, &ds) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_dataset_create(3, 2, 1, NULL) == NN_ERR_NULL_ARG);
}

static void test_free_null_is_noop(void)
{
    nn_dataset_free(NULL);
    NN_ASSERT(1);
}

static void test_load_csv_round_trip(void)
{
    FILE *f = fopen(NN_TEST_CSV_PATH, "w");
    NN_ASSERT(f != NULL);
    fprintf(f, "0,0,0\n");
    fprintf(f, "\n"); /* blank line, must be skipped */
    fprintf(f, "0,1,1\n");
    fprintf(f, "1,0,1\n");
    fprintf(f, "1,1,0\n");
    fclose(f);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_load_csv(NN_TEST_CSV_PATH, 2, 1, &ds) == NN_OK);
    NN_ASSERT(ds->n_samples == 4);

    NN_ASSERT_NEAR(ds->inputs[0 * 2 + 0], 0.0, 1e-12);
    NN_ASSERT_NEAR(ds->inputs[0 * 2 + 1], 0.0, 1e-12);
    NN_ASSERT_NEAR(ds->targets[0], 0.0, 1e-12);

    NN_ASSERT_NEAR(ds->inputs[2 * 2 + 0], 1.0, 1e-12);
    NN_ASSERT_NEAR(ds->inputs[2 * 2 + 1], 0.0, 1e-12);
    NN_ASSERT_NEAR(ds->targets[2], 1.0, 1e-12);

    nn_dataset_free(ds);
    remove(NN_TEST_CSV_PATH);
}

static void test_load_csv_missing_file_is_io_error(void)
{
    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_load_csv("build/does_not_exist.csv", 2, 1, &ds) == NN_ERR_IO);
}

static void test_load_csv_malformed_row_is_format_error(void)
{
    const char *path = "build/nn_test_dataset_malformed.csv";
    FILE *f = fopen(path, "w");
    NN_ASSERT(f != NULL);
    fprintf(f, "0,0,0\n");
    fprintf(f, "1,1\n"); /* missing one field */
    fclose(f);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_load_csv(path, 2, 1, &ds) == NN_ERR_FORMAT);

    remove(path);
}

int main(void)
{
    NN_RUN(test_create_and_free);
    NN_RUN(test_create_rejects_bad_args);
    NN_RUN(test_free_null_is_noop);
    NN_RUN(test_load_csv_round_trip);
    NN_RUN(test_load_csv_missing_file_is_io_error);
    NN_RUN(test_load_csv_malformed_row_is_format_error);
    return NN_TEST_RESULT();
}
