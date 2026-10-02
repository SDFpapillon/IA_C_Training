#include "nn.h"
#include "test.h"

static void test_linear(void)
{
    NN_ASSERT_NEAR(nn_activation_apply(NN_ACT_LINEAR, 3.5), 3.5, 1e-12);
    NN_ASSERT_NEAR(nn_activation_derivative(NN_ACT_LINEAR, 3.5), 1.0, 1e-12);
}

static void test_sigmoid(void)
{
    NN_ASSERT_NEAR(nn_activation_apply(NN_ACT_SIGMOID, 0.0), 0.5, 1e-12);
    NN_ASSERT_NEAR(nn_activation_derivative(NN_ACT_SIGMOID, 0.0), 0.25, 1e-12);

    nn_real y = nn_activation_apply(NN_ACT_SIGMOID, 10.0);
    NN_ASSERT(y > 0.99 && y < 1.0);
}

static void test_tanh_activation(void)
{
    NN_ASSERT_NEAR(nn_activation_apply(NN_ACT_TANH, 0.0), 0.0, 1e-12);
    NN_ASSERT_NEAR(nn_activation_derivative(NN_ACT_TANH, 0.0), 1.0, 1e-12);
}

static void test_relu(void)
{
    NN_ASSERT_NEAR(nn_activation_apply(NN_ACT_RELU, -2.0), 0.0, 1e-12);
    NN_ASSERT_NEAR(nn_activation_apply(NN_ACT_RELU, 2.0), 2.0, 1e-12);
    NN_ASSERT_NEAR(nn_activation_derivative(NN_ACT_RELU, -2.0), 0.0, 1e-12);
    NN_ASSERT_NEAR(nn_activation_derivative(NN_ACT_RELU, 2.0), 1.0, 1e-12);
}

/* Finite-difference check: derivative should match the local slope, away
 * from ReLU's kink at 0. */
static void test_derivative_matches_finite_difference(void)
{
    static const nn_activation acts[] = {NN_ACT_LINEAR, NN_ACT_SIGMOID, NN_ACT_TANH, NN_ACT_RELU};
    static const nn_real points[] = {-3.0, -1.0, 1.0, 3.0};
    const nn_real h = 1e-6;

    for (size_t a = 0; a < sizeof(acts) / sizeof(acts[0]); a++) {
        for (size_t p = 0; p < sizeof(points) / sizeof(points[0]); p++) {
            nn_real x = points[p];
            nn_real numeric = (nn_activation_apply(acts[a], x + h) -
                                nn_activation_apply(acts[a], x - h)) / (2.0 * h);
            nn_real analytic = nn_activation_derivative(acts[a], x);
            NN_ASSERT_NEAR(numeric, analytic, 1e-4);
        }
    }
}

int main(void)
{
    NN_RUN(test_linear);
    NN_RUN(test_sigmoid);
    NN_RUN(test_tanh_activation);
    NN_RUN(test_relu);
    NN_RUN(test_derivative_matches_finite_difference);
    return NN_TEST_RESULT();
}
