#include <math.h>

#include "nn.h"
#include "test.h"

/* These networks are built by hand (not via nn_create) so the expected
 * outputs can be computed by hand too. Their weights/biases live on the
 * stack, so nn_free() must not be called on them. */

static void test_forward_single_linear_layer(void)
{
    nn_real weights[] = {
        1.0, 2.0, /* output 0 = 1*x0 + 2*x1 */
        0.0, 1.0, /* output 1 = 0*x0 + 1*x1 */
    };
    nn_real biases[] = {0.5, -1.0};
    nn_layer layer = {2, 2, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real input[] = {3.0, 4.0};
    nn_real output[2];
    NN_ASSERT(nn_forward(&net, buf, input, output) == NN_OK);

    NN_ASSERT_NEAR(output[0], 1.0 * 3.0 + 2.0 * 4.0 + 0.5, 1e-12);
    NN_ASSERT_NEAR(output[1], 0.0 * 3.0 + 1.0 * 4.0 - 1.0, 1e-12);

    nn_forward_buffer_free(buf);
}

static void test_forward_sigmoid_at_zero(void)
{
    nn_real weights[] = {1.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_SIGMOID, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real input[] = {0.0};
    nn_real output[1];
    NN_ASSERT(nn_forward(&net, buf, input, output) == NN_OK);
    NN_ASSERT_NEAR(output[0], 0.5, 1e-12);

    nn_forward_buffer_free(buf);
}

static void test_forward_chains_multiple_layers(void)
{
    /* Layer 0: identity (2 -> 2, linear). Layer 1: sum (2 -> 1, linear). */
    nn_real w0[] = {1.0, 0.0, 0.0, 1.0};
    nn_real b0[] = {0.0, 0.0};
    nn_real w1[] = {1.0, 1.0};
    nn_real b1[] = {0.0};
    nn_layer layers[] = {
        {2, 2, NN_ACT_LINEAR, w0, b0},
        {2, 1, NN_ACT_LINEAR, w1, b1},
    };
    nn_network net = {2, layers};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real input[] = {3.0, 4.0};
    nn_real output[1];
    NN_ASSERT(nn_forward(&net, buf, input, output) == NN_OK);
    NN_ASSERT_NEAR(output[0], 7.0, 1e-12);

    nn_forward_buffer_free(buf);
}

static void test_forward_softmax_known_logits(void)
{
    /* Zero weights -> logits equal the biases regardless of input. */
    nn_real weights[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    nn_real biases[] = {1.0, 2.0, 3.0};
    nn_layer layer = {2, 3, NN_ACT_SOFTMAX, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real input[] = {10.0, -10.0}; /* irrelevant: weights are zero */
    nn_real output[3];
    NN_ASSERT(nn_forward(&net, buf, input, output) == NN_OK);

    double sum_exp = exp(1.0) + exp(2.0) + exp(3.0);
    NN_ASSERT_NEAR(output[0], exp(1.0) / sum_exp, 1e-12);
    NN_ASSERT_NEAR(output[1], exp(2.0) / sum_exp, 1e-12);
    NN_ASSERT_NEAR(output[2], exp(3.0) / sum_exp, 1e-12);
    NN_ASSERT_NEAR(output[0] + output[1] + output[2], 1.0, 1e-12);

    nn_forward_buffer_free(buf);
}

static void test_forward_buffer_create_rejects_null(void)
{
    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(NULL, &buf) == NN_ERR_NULL_ARG);

    nn_real weights[] = {1.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};
    NN_ASSERT(nn_forward_buffer_create(&net, NULL) == NN_ERR_NULL_ARG);
}

static void test_forward_rejects_mismatched_buffer(void)
{
    nn_real w1[] = {1.0};
    nn_real b1[] = {0.0};
    nn_layer layer1 = {1, 1, NN_ACT_LINEAR, w1, b1};
    nn_network net1 = {1, &layer1};

    nn_real w2a[] = {1.0};
    nn_real b2a[] = {0.0};
    nn_real w2b[] = {1.0};
    nn_real b2b[] = {0.0};
    nn_layer layers2[] = {
        {1, 1, NN_ACT_LINEAR, w2a, b2a},
        {1, 1, NN_ACT_LINEAR, w2b, b2b},
    };
    nn_network net2 = {2, layers2};

    nn_forward_buffer *buf2 = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net2, &buf2) == NN_OK);

    nn_real input[] = {1.0};
    nn_real output[1];
    NN_ASSERT(nn_forward(&net1, buf2, input, output) == NN_ERR_SIZE_MISMATCH);

    nn_forward_buffer_free(buf2);
}

static void test_forward_buffer_free_null_is_noop(void)
{
    nn_forward_buffer_free(NULL);
    NN_ASSERT(1); /* reaching here without crashing is the test */
}

static void test_forward_buffer_is_reused_across_calls(void)
{
    nn_real weights[] = {2.0};
    nn_real biases[] = {0.0};
    nn_layer layer = {1, 1, NN_ACT_LINEAR, weights, biases};
    nn_network net = {1, &layer};

    nn_forward_buffer *buf = NULL;
    NN_ASSERT(nn_forward_buffer_create(&net, &buf) == NN_OK);

    nn_real output[1];

    nn_real input_a[] = {3.0};
    NN_ASSERT(nn_forward(&net, buf, input_a, output) == NN_OK);
    NN_ASSERT_NEAR(output[0], 6.0, 1e-12);

    nn_real input_b[] = {5.0};
    NN_ASSERT(nn_forward(&net, buf, input_b, output) == NN_OK);
    NN_ASSERT_NEAR(output[0], 10.0, 1e-12);

    nn_forward_buffer_free(buf);
}

int main(void)
{
    NN_RUN(test_forward_single_linear_layer);
    NN_RUN(test_forward_sigmoid_at_zero);
    NN_RUN(test_forward_chains_multiple_layers);
    NN_RUN(test_forward_softmax_known_logits);
    NN_RUN(test_forward_buffer_create_rejects_null);
    NN_RUN(test_forward_rejects_mismatched_buffer);
    NN_RUN(test_forward_buffer_free_null_is_noop);
    NN_RUN(test_forward_buffer_is_reused_across_calls);
    return NN_TEST_RESULT();
}
