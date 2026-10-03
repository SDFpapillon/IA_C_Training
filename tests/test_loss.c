#include <math.h>

#include "nn.h"
#include "test.h"

static void test_mse_zero_when_equal(void)
{
    nn_real output[] = {1.0, 2.0};
    nn_real target[] = {1.0, 2.0};
    NN_ASSERT_NEAR(nn_loss_value(NN_LOSS_MSE, output, target, 2), 0.0, 1e-12);
}

static void test_mse_known_value(void)
{
    nn_real output[] = {0.0, 0.0};
    nn_real target[] = {1.0, 1.0};
    /* ((0-1)^2 + (0-1)^2) / 2 = 1.0 */
    NN_ASSERT_NEAR(nn_loss_value(NN_LOSS_MSE, output, target, 2), 1.0, 1e-12);
}

static void test_cross_entropy_known_value(void)
{
    nn_real output[] = {0.7, 0.2, 0.1};
    nn_real target[] = {1.0, 0.0, 0.0};
    NN_ASSERT_NEAR(nn_loss_value(NN_LOSS_CROSS_ENTROPY, output, target, 3), -log(0.7), 1e-12);
}

int main(void)
{
    NN_RUN(test_mse_zero_when_equal);
    NN_RUN(test_mse_known_value);
    NN_RUN(test_cross_entropy_known_value);
    return NN_TEST_RESULT();
}
