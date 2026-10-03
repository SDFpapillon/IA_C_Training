#include <stdio.h>
#include <string.h>

#include "nn.h"
#include "test.h"

#define NN_TEST_PRED_PATH "build/nn_test_export_predictions.csv"
#define NN_TEST_PPM_PATH "build/nn_test_export_boundary.ppm"

static void test_export_predictions_csv_known_values(void)
{
    nn_real weights[] = {2.0}; /* output = 2 * x */
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 1, 1, &ds) == NN_OK);
    ds->inputs[0] = 1.0;
    ds->targets[0] = 3.0; /* predicted = 2.0 */
    ds->inputs[1] = 2.0;
    ds->targets[1] = 5.0; /* predicted = 4.0 */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    NN_ASSERT(nn_export_predictions_csv(&net, buf, ds, NN_TEST_PRED_PATH) == NN_OK);

    FILE *f = fopen(NN_TEST_PRED_PATH, "r");
    NN_ASSERT(f != NULL);
    char header[128];
    NN_ASSERT(fgets(header, sizeof(header), f) != NULL);
    NN_ASSERT(strncmp(header, "sample,input_0,expected_0,predicted_0",
                       strlen("sample,input_0,expected_0,predicted_0")) == 0);

    size_t sample;
    double input0, expected0, predicted0;
    NN_ASSERT(fscanf(f, "%zu,%lf,%lf,%lf\n", &sample, &input0, &expected0, &predicted0) == 4);
    NN_ASSERT(sample == 0);
    NN_ASSERT_NEAR(input0, 1.0, 1e-12);
    NN_ASSERT_NEAR(expected0, 3.0, 1e-12);
    NN_ASSERT_NEAR(predicted0, 2.0, 1e-12);

    NN_ASSERT(fscanf(f, "%zu,%lf,%lf,%lf\n", &sample, &input0, &expected0, &predicted0) == 4);
    NN_ASSERT(sample == 1);
    NN_ASSERT_NEAR(input0, 2.0, 1e-12);
    NN_ASSERT_NEAR(expected0, 5.0, 1e-12);
    NN_ASSERT_NEAR(predicted0, 4.0, 1e-12);

    fclose(f);
    remove(NN_TEST_PRED_PATH);
    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

static void test_export_predictions_rejects_dimension_mismatch(void)
{
    nn_real weights[] = {1.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(1, 2, 1, &ds) == NN_OK); /* wrong n_inputs */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    NN_ASSERT(nn_export_predictions_csv(&net, buf, ds, NN_TEST_PRED_PATH) == NN_ERR_SIZE_MISMATCH);

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

static int read_ppm_header(FILE *f, int *width, int *height, int *maxval)
{
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        return 0;
    }
    if (fscanf(f, "%d %d %d", width, height, maxval) != 3) {
        return 0;
    }
    fgetc(f); /* single whitespace byte separating the header from pixel data */
    return 1;
}

static void test_export_decision_boundary_ppm_grayscale(void)
{
    /* output = axis_x value - 0.5 (axis_y's weight is 0: it must not affect the result). */
    nn_real weights[] = {1.0, 0.0};
    nn_real biases[] = {-0.5};
    nn_layer layer = {2, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real fixed_inputs[] = {0.0, 0.0};
    NN_ASSERT(nn_export_decision_boundary_ppm(&net, buf, fixed_inputs, 0, 1, 0.0, 1.0, 0.0, 1.0, 2, 1,
                                               NN_TEST_PPM_PATH) == NN_OK);

    FILE *f = fopen(NN_TEST_PPM_PATH, "rb");
    NN_ASSERT(f != NULL);
    int width = 0, height = 0, maxval = 0;
    NN_ASSERT(read_ppm_header(f, &width, &height, &maxval));
    NN_ASSERT(width == 2 && height == 1 && maxval == 255);

    unsigned char pixels[6];
    NN_ASSERT(fread(pixels, 1, 6, f) == 6);
    /* col 0: x=0 -> output=-0.5 clamped to 0 -> gray 0 */
    NN_ASSERT(pixels[0] == 0 && pixels[1] == 0 && pixels[2] == 0);
    /* col 1: x=1 -> output=0.5 -> gray (unsigned char)(0.5*255) = 127 */
    NN_ASSERT(pixels[3] == 127 && pixels[4] == 127 && pixels[5] == 127);

    fclose(f);
    remove(NN_TEST_PPM_PATH);
    nn_forward_buffer_free(buf);
}

static void test_export_decision_boundary_ppm_argmax_palette(void)
{
    /* 2 outputs, softmax: class 0 wins when x<0.5, class 1 wins when x>0.5. */
    nn_real weights[] = {-1.0, 0.0, 1.0, 0.0};
    nn_real biases[] = {0.0, 0.0};
    nn_layer layer = {2, 2, NN_ACT_SOFTMAX, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real fixed_inputs[] = {0.0, 0.0};
    NN_ASSERT(nn_export_decision_boundary_ppm(&net, buf, fixed_inputs, 0, 1, -1.0, 1.0, 0.0, 1.0, 2, 1,
                                               NN_TEST_PPM_PATH) == NN_OK);

    FILE *f = fopen(NN_TEST_PPM_PATH, "rb");
    NN_ASSERT(f != NULL);
    int width = 0, height = 0, maxval = 0;
    NN_ASSERT(read_ppm_header(f, &width, &height, &maxval));

    unsigned char pixels[6];
    NN_ASSERT(fread(pixels, 1, 6, f) == 6);
    /* col 0: x=-1 -> class 0's logit (1.0) > class 1's (-1.0) -> palette[0] */
    NN_ASSERT(pixels[0] == 230 && pixels[1] == 25 && pixels[2] == 75);
    /* col 1: x=1 -> class 1's logit (1.0) > class 0's (-1.0) -> palette[1] */
    NN_ASSERT(pixels[3] == 60 && pixels[4] == 180 && pixels[5] == 75);

    fclose(f);
    remove(NN_TEST_PPM_PATH);
    nn_forward_buffer_free(buf);
}

static void test_export_decision_boundary_rejects_bad_axes(void)
{
    nn_real weights[] = {1.0, 0.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {2, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real fixed_inputs[] = {0.0, 0.0};

    NN_ASSERT(nn_export_decision_boundary_ppm(&net, buf, fixed_inputs, 0, 0, 0.0, 1.0, 0.0, 1.0, 2, 2,
                                               NN_TEST_PPM_PATH) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_export_decision_boundary_ppm(&net, buf, fixed_inputs, 0, 2, 0.0, 1.0, 0.0, 1.0, 2, 2,
                                               NN_TEST_PPM_PATH) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_export_decision_boundary_ppm(&net, buf, fixed_inputs, 0, 1, 0.0, 1.0, 0.0, 1.0, 0, 2,
                                               NN_TEST_PPM_PATH) == NN_ERR_INVALID_ARG);

    nn_forward_buffer_free(buf);
}

int main(void)
{
    NN_RUN(test_export_predictions_csv_known_values);
    NN_RUN(test_export_predictions_rejects_dimension_mismatch);
    NN_RUN(test_export_decision_boundary_ppm_grayscale);
    NN_RUN(test_export_decision_boundary_ppm_argmax_palette);
    NN_RUN(test_export_decision_boundary_rejects_bad_axes);
    return NN_TEST_RESULT();
}
