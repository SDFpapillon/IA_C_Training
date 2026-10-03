#include "nn.h"
#include "test.h"

static void test_accuracy_threshold_single_output(void)
{
    /* Weights zero: output is constant regardless of input. bias=2 -> sigmoid(2) > 0.5. */
    nn_real weights[] = {0.0};
    nn_real biases[] = {2.0};
    nn_layer layer = {1, 1, NN_ACT_SIGMOID, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 1, 1, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->targets[0] = 1.0; /* output > 0.5: correct */
    ds->inputs[1] = 0.0;
    ds->targets[1] = 0.0; /* output > 0.5: incorrect */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real output[1];

    nn_real accuracy = -1.0;
    NN_ASSERT(nn_dataset_accuracy(&net, buf, output, ds, &accuracy) == NN_OK);
    NN_ASSERT_NEAR(accuracy, 0.5, 1e-12);

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

static void test_accuracy_argmax_multi_output(void)
{
    /* Weights zero: logits are the biases regardless of input; softmax's
     * argmax is always class index 2 (the largest bias). */
    nn_real weights[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    nn_real biases[] = {1.0, 2.0, 3.0};
    nn_layer layer = {2, 3, NN_ACT_SOFTMAX, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 2, 3, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->inputs[1] = 0.0;
    ds->targets[0] = 0.0;
    ds->targets[1] = 0.0;
    ds->targets[2] = 1.0; /* one-hot class 2: matches the network's argmax -> correct */

    ds->inputs[2] = 1.0;
    ds->inputs[3] = 1.0;
    ds->targets[3] = 1.0;
    ds->targets[4] = 0.0;
    ds->targets[5] = 0.0; /* one-hot class 0: does not match -> incorrect */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real output[3];

    nn_real accuracy = -1.0;
    NN_ASSERT(nn_dataset_accuracy(&net, buf, output, ds, &accuracy) == NN_OK);
    NN_ASSERT_NEAR(accuracy, 0.5, 1e-12);

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

static void test_loss_known_mse_average(void)
{
    /* Weights and bias zero: output is always 0. */
    nn_real weights[] = {0.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 1, 1, &ds) == NN_OK);
    ds->inputs[0] = 0.0;
    ds->targets[0] = 1.0; /* (0-1)^2 = 1 */
    ds->inputs[1] = 0.0;
    ds->targets[1] = 0.5; /* (0-0.5)^2 = 0.25 */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real output[1];

    nn_real loss = -1.0;
    NN_ASSERT(nn_dataset_loss(&net, buf, output, ds, NN_LOSS_MSE, &loss) == NN_OK);
    NN_ASSERT_NEAR(loss, (1.0 + 0.25) / 2.0, 1e-12);

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

static void test_rejects_dimension_mismatch(void)
{
    nn_real weights[] = {0.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 2, 1, &ds) == NN_OK); /* wrong n_inputs (2 vs net's 1) */

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);
    nn_real output[1];
    nn_real result = -1.0;

    NN_ASSERT(nn_dataset_loss(&net, buf, output, ds, NN_LOSS_MSE, &result) == NN_ERR_SIZE_MISMATCH);
    NN_ASSERT(nn_dataset_accuracy(&net, buf, output, ds, &result) == NN_ERR_SIZE_MISMATCH);

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
}

int main(void)
{
    NN_RUN(test_accuracy_threshold_single_output);
    NN_RUN(test_accuracy_argmax_multi_output);
    NN_RUN(test_loss_known_mse_average);
    NN_RUN(test_rejects_dimension_mismatch);
    return NN_TEST_RESULT();
}
