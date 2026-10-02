#include "nn.h"
#include "test.h"

static void test_rng_deterministic_same_seed(void)
{
    nn_rng a, b;
    nn_rng_seed(&a, 42);
    nn_rng_seed(&b, 42);

    for (int i = 0; i < 100; i++) {
        NN_ASSERT(nn_rng_next_u64(&a) == nn_rng_next_u64(&b));
    }
}

static void test_rng_different_seeds_diverge(void)
{
    nn_rng a, b;
    nn_rng_seed(&a, 1);
    nn_rng_seed(&b, 2);

    NN_ASSERT(nn_rng_next_u64(&a) != nn_rng_next_u64(&b));
}

static void test_rng_uniform_in_range(void)
{
    nn_rng rng;
    nn_rng_seed(&rng, 7);

    for (int i = 0; i < 1000; i++) {
        nn_real x = nn_rng_uniform(&rng);
        NN_ASSERT(x >= 0.0 && x < 1.0);
    }
}

static void test_rng_uniform_range_bounds(void)
{
    nn_rng rng;
    nn_rng_seed(&rng, 123);

    for (int i = 0; i < 1000; i++) {
        nn_real x = nn_rng_uniform_range(&rng, -2.0, 3.0);
        NN_ASSERT(x >= -2.0 && x < 3.0);
    }
}

static void test_rng_normal_is_finite_and_bounded(void)
{
    nn_rng rng;
    nn_rng_seed(&rng, 99);

    for (int i = 0; i < 1000; i++) {
        nn_real x = nn_rng_normal(&rng);
        /* 1000 draws landing outside [-8, 8] sigma would indicate a bug. */
        NN_ASSERT(x > -8.0 && x < 8.0);
    }
}

int main(void)
{
    NN_RUN(test_rng_deterministic_same_seed);
    NN_RUN(test_rng_different_seeds_diverge);
    NN_RUN(test_rng_uniform_in_range);
    NN_RUN(test_rng_uniform_range_bounds);
    NN_RUN(test_rng_normal_is_finite_and_bounded);
    return NN_TEST_RESULT();
}
