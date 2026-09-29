/*
 * Minimal test helpers — no dependency.
 *
 *   static void test_something(void) { NN_ASSERT(1 + 1 == 2); }
 *   int main(void) { NN_RUN(test_something); return NN_TEST_RESULT(); }
 */
#ifndef NN_TEST_H
#define NN_TEST_H

#include <math.h>
#include <stdio.h>

static int nn_test_failures = 0;
static int nn_test_count = 0;

#define NN_ASSERT(cond)                                                     \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            nn_test_failures++;                                             \
        }                                                                   \
    } while (0)

#define NN_ASSERT_NEAR(a, b, eps) NN_ASSERT(fabs((double)(a) - (double)(b)) <= (eps))

#define NN_RUN(fn)                     \
    do {                               \
        int before_ = nn_test_failures; \
        nn_test_count++;               \
        fn();                          \
        printf("%s %s\n", nn_test_failures == before_ ? "  ok  " : "  FAIL", #fn); \
    } while (0)

#define NN_TEST_RESULT() \
    (printf("%d test(s), %d failure(s)\n", nn_test_count, nn_test_failures), nn_test_failures != 0)

#endif /* NN_TEST_H */
