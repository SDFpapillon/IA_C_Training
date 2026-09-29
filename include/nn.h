/*
 * Ia_c_training — public API
 *
 * Conventions:
 *   - Every public symbol is prefixed with `nn_` (types, functions) or `NN_` (macros, enum values).
 *   - All numeric values use `nn_real` (double).
 *   - Functions that can fail return an `nn_status`; results are written through out-parameters.
 *   - Objects are created with nn_*_create() and released with the matching nn_*_free().
 *     nn_*_free(NULL) is always a no-op.
 */
#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#define NN_VERSION_MAJOR 0
#define NN_VERSION_MINOR 1
#define NN_VERSION_PATCH 0

/* Floating-point type used for weights, activations and gradients. */
typedef double nn_real;

typedef enum nn_status {
    NN_OK = 0,
    NN_ERR_NULL_ARG,      /* a required pointer argument was NULL */
    NN_ERR_INVALID_ARG,   /* an argument has an invalid value (size 0, bad enum, ...) */
    NN_ERR_ALLOC,         /* memory allocation failed */
    NN_ERR_SIZE_MISMATCH, /* input/output/dataset dimensions do not match the network */
    NN_ERR_IO,            /* file could not be opened, read or written */
    NN_ERR_FORMAT         /* file content is malformed */
} nn_status;

/* Human-readable, static description of a status code. Never returns NULL. */
const char *nn_status_str(nn_status status);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */
