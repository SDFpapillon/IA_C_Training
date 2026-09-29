#include <string.h>

#include "nn.h"
#include "test.h"

static void test_status_str_known(void)
{
    NN_ASSERT(strcmp(nn_status_str(NN_OK), "ok") == 0);
    NN_ASSERT(strcmp(nn_status_str(NN_ERR_ALLOC), "memory allocation failed") == 0);
}

static void test_status_str_unknown_never_null(void)
{
    NN_ASSERT(nn_status_str((nn_status)999) != NULL);
}

static void test_real_is_double(void)
{
    NN_ASSERT(sizeof(nn_real) == sizeof(double));
}

int main(void)
{
    NN_RUN(test_status_str_known);
    NN_RUN(test_status_str_unknown_never_null);
    NN_RUN(test_real_is_double);
    return NN_TEST_RESULT();
}
