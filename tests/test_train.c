#include "nn.h"
#include "test.h"

static nn_real average_loss(const nn_network *net, const nn_dataset *ds)
{
    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(net, &buf) == NN_OK);

    nn_real total = 0.0;
    for (size_t s = 0; s < ds->n_samples; s++) {
        nn_real output[1];
        const nn_real *in = &ds->inputs[s * ds->n_inputs];
        const nn_real *target = &ds->targets[s * ds->n_outputs];
        NN_ASSERT(nn_forward(net, buf, in, output) == NN_OK);
        total += nn_loss_value(NN_LOSS_MSE, output, target, ds->n_outputs);
    }

    nn_forward_buffer_free(buf);
    return total / (nn_real)ds->n_samples;
}

static void test_train_supervised_learns_xor(void)
{
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &net) == NN_OK);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(4, 2, 1, &ds) == NN_OK);
    nn_real inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    nn_real targets[] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; i++) {
        ds->inputs[i] = inputs[i];
    }
    for (size_t i = 0; i < 4; i++) {
        ds->targets[i] = targets[i];
    }

    nn_real loss_before = average_loss(net, ds);

    nn_train_params params = {0};
    params.learning_rate = 0.5;
    params.epochs = 5000;
    params.batch_size = 0; /* full batch */
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 123;

    NN_ASSERT(nn_train_supervised(net, ds, &params) == NN_OK);

    nn_real loss_after = average_loss(net, ds);
    NN_ASSERT(loss_after < loss_before);
    NN_ASSERT(loss_after < 0.02);

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(net, &buf) == NN_OK);
    for (size_t s = 0; s < ds->n_samples; s++) {
        nn_real output[1];
        NN_ASSERT(nn_forward(net, buf, &ds->inputs[s * 2], output) == NN_OK);
        nn_real expected = ds->targets[s];
        if (expected > 0.5) {
            NN_ASSERT(output[0] > 0.5);
        } else {
            NN_ASSERT(output[0] < 0.5);
        }
    }
    nn_forward_buffer_free(buf);

    nn_dataset_free(ds);
    nn_free(net);
}

static void test_train_supervised_rejects_dimension_mismatch(void)
{
    size_t sizes[] = {2, 1};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 2, acts, 1, NN_INIT_UNIFORM, 1, &net) == NN_OK);

    nn_dataset *ds = NULL;
    NN_ASSERT(nn_dataset_create(2, 3, 1, &ds) == NN_OK); /* wrong n_inputs */

    nn_train_params params = {0};
    params.learning_rate = 0.1;
    params.epochs = 1;
    params.loss = NN_LOSS_MSE;

    NN_ASSERT(nn_train_supervised(net, ds, &params) == NN_ERR_SIZE_MISMATCH);

    nn_dataset_free(ds);
    nn_free(net);
}

int main(void)
{
    NN_RUN(test_train_supervised_learns_xor);
    NN_RUN(test_train_supervised_rejects_dimension_mismatch);
    return NN_TEST_RESULT();
}
