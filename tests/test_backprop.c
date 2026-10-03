#include "nn.h"
#include "test.h"

/* Loss of net on (input, target) under the current weights, via a plain
 * forward pass (no gradient side effects). */
static nn_real forward_loss(const nn_network *net, nn_forward_buffer *fbuf,
                             const nn_real *input, const nn_real *target,
                             nn_loss loss, size_t n_out)
{
    nn_real output[8];
    NN_ASSERT(nn_forward(net, fbuf, input, output) == NN_OK);
    return nn_loss_value(loss, output, target, n_out);
}

/* Compare nn_backprop()'s analytic gradient against a central finite
 * difference on every weight and bias, by nudging each parameter in turn
 * and re-running the forward pass. */
static void check_gradient(nn_network *net, const nn_real *input, const nn_real *target,
                            nn_loss loss)
{
    nn_forward_buffer *fbuf = NULL;
    nn_backprop_buffer *bbuf = NULL;
    nn_gradient *grad = NULL;
    NN_ASSERT(nn_forward_buffer_create(net, &fbuf) == NN_OK);
    NN_ASSERT(nn_backprop_buffer_create(net, &bbuf) == NN_OK);
    NN_ASSERT(nn_gradient_create(net, &grad) == NN_OK);

    size_t n_out = net->layers[net->n_layers - 1].n_outputs;
    NN_ASSERT(nn_backprop(net, bbuf, input, target, loss, grad) == NN_OK);

    const nn_real h = 1e-5;
    const nn_real tol = 1e-4;

    for (size_t l = 0; l < net->n_layers; l++) {
        nn_layer *layer = &net->layers[l];
        size_t n_w = layer->n_inputs * layer->n_outputs;

        for (size_t w = 0; w < n_w; w++) {
            nn_real original = layer->weights[w];

            layer->weights[w] = original + h;
            nn_real loss_plus = forward_loss(net, fbuf, input, target, loss, n_out);

            layer->weights[w] = original - h;
            nn_real loss_minus = forward_loss(net, fbuf, input, target, loss, n_out);

            layer->weights[w] = original;

            nn_real numeric = (loss_plus - loss_minus) / (2.0 * h);
            NN_ASSERT_NEAR(numeric, grad->dweights[l][w], tol);
        }

        for (size_t o = 0; o < layer->n_outputs; o++) {
            nn_real original = layer->biases[o];

            layer->biases[o] = original + h;
            nn_real loss_plus = forward_loss(net, fbuf, input, target, loss, n_out);

            layer->biases[o] = original - h;
            nn_real loss_minus = forward_loss(net, fbuf, input, target, loss, n_out);

            layer->biases[o] = original;

            nn_real numeric = (loss_plus - loss_minus) / (2.0 * h);
            NN_ASSERT_NEAR(numeric, grad->dbiases[l][o], tol);
        }
    }

    nn_gradient_free(grad);
    nn_backprop_buffer_free(bbuf);
    nn_forward_buffer_free(fbuf);
}

static void test_gradient_matches_finite_difference_mse(void)
{
    size_t sizes[] = {3, 4, 2};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 42, &net) == NN_OK);

    nn_real input[] = {0.5, -0.2, 0.1};
    nn_real target[] = {0.3, 0.8};
    check_gradient(net, input, target, NN_LOSS_MSE);

    nn_free(net);
}

static void test_gradient_matches_finite_difference_softmax_cross_entropy(void)
{
    size_t sizes[] = {3, 4, 3};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SOFTMAX};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 7, &net) == NN_OK);

    nn_real input[] = {0.2, 0.4, -0.3};
    nn_real target[] = {0.0, 1.0, 0.0}; /* one-hot */
    check_gradient(net, input, target, NN_LOSS_CROSS_ENTROPY);

    nn_free(net);
}

static void test_gradient_matches_finite_difference_relu_hidden(void)
{
    /* Biases offset so ReLU units are active (non-zero slope) at this input. */
    size_t sizes[] = {2, 3, 1};
    nn_activation acts[] = {NN_ACT_RELU, NN_ACT_LINEAR};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 3, acts, 2, NN_INIT_HE, 11, &net) == NN_OK);
    for (size_t o = 0; o < net->layers[0].n_outputs; o++) {
        net->layers[0].biases[o] = 1.0;
    }

    nn_real input[] = {0.3, 0.6};
    nn_real target[] = {1.5};
    check_gradient(net, input, target, NN_LOSS_MSE);

    nn_free(net);
}

static void test_backprop_rejects_incompatible_loss_activation(void)
{
    size_t sizes[] = {2, 2};
    nn_activation acts_softmax[] = {NN_ACT_SOFTMAX};
    nn_activation acts_linear[] = {NN_ACT_LINEAR};
    nn_network *net_softmax = NULL;
    nn_network *net_linear = NULL;
    NN_ASSERT(nn_create(sizes, 2, acts_softmax, 1, NN_INIT_UNIFORM, 1, &net_softmax) == NN_OK);
    NN_ASSERT(nn_create(sizes, 2, acts_linear, 1, NN_INIT_UNIFORM, 1, &net_linear) == NN_OK);

    nn_backprop_buffer *buf_s = NULL;
    nn_backprop_buffer *buf_l = NULL;
    nn_gradient *grad_s = NULL;
    nn_gradient *grad_l = NULL;
    NN_ASSERT(nn_backprop_buffer_create(net_softmax, &buf_s) == NN_OK);
    NN_ASSERT(nn_backprop_buffer_create(net_linear, &buf_l) == NN_OK);
    NN_ASSERT(nn_gradient_create(net_softmax, &grad_s) == NN_OK);
    NN_ASSERT(nn_gradient_create(net_linear, &grad_l) == NN_OK);

    nn_real input[] = {0.1, 0.2};
    nn_real target[] = {1.0, 0.0};

    NN_ASSERT(nn_backprop(net_softmax, buf_s, input, target, NN_LOSS_MSE, grad_s) == NN_ERR_INVALID_ARG);
    NN_ASSERT(nn_backprop(net_linear, buf_l, input, target, NN_LOSS_CROSS_ENTROPY, grad_l) == NN_ERR_INVALID_ARG);

    nn_gradient_free(grad_s);
    nn_gradient_free(grad_l);
    nn_backprop_buffer_free(buf_s);
    nn_backprop_buffer_free(buf_l);
    nn_free(net_softmax);
    nn_free(net_linear);
}

static void test_gradient_zero_resets_accumulation(void)
{
    size_t sizes[] = {2, 2};
    nn_activation acts[] = {NN_ACT_LINEAR};
    nn_network *net = NULL;
    NN_ASSERT(nn_create(sizes, 2, acts, 1, NN_INIT_UNIFORM, 1, &net) == NN_OK);

    nn_backprop_buffer *buf = NULL;
    nn_gradient *grad = NULL;
    NN_ASSERT(nn_backprop_buffer_create(net, &buf) == NN_OK);
    NN_ASSERT(nn_gradient_create(net, &grad) == NN_OK);

    nn_real input[] = {0.1, 0.2};
    nn_real target[] = {1.0, 0.0};
    NN_ASSERT(nn_backprop(net, buf, input, target, NN_LOSS_MSE, grad) == NN_OK);
    NN_ASSERT(grad->dweights[0][0] != 0.0);

    nn_gradient_zero(grad);
    for (size_t w = 0; w < net->layers[0].n_inputs * net->layers[0].n_outputs; w++) {
        NN_ASSERT(grad->dweights[0][w] == 0.0);
    }

    nn_gradient_free(grad);
    nn_backprop_buffer_free(buf);
    nn_free(net);
}

int main(void)
{
    NN_RUN(test_gradient_matches_finite_difference_mse);
    NN_RUN(test_gradient_matches_finite_difference_softmax_cross_entropy);
    NN_RUN(test_gradient_matches_finite_difference_relu_hidden);
    NN_RUN(test_backprop_rejects_incompatible_loss_activation);
    NN_RUN(test_gradient_zero_resets_accumulation);
    return NN_TEST_RESULT();
}
