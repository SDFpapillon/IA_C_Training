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

#include <stddef.h>
#include <stdint.h>

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

/* --------------------------------------------------------------------- */
/* RNG                                                                    */
/* --------------------------------------------------------------------- */

/*
 * Portable, reproducible RNG (splitmix64). Does not rely on the C library
 * rand(), whose sequence and quality differ across platforms/libc versions.
 * Same seed -> same sequence, on every platform.
 */
typedef struct nn_rng {
    uint64_t state;
} nn_rng;

void nn_rng_seed(nn_rng *rng, uint64_t seed);

/* Next raw 64-bit value. */
uint64_t nn_rng_next_u64(nn_rng *rng);

/* Uniform double in [0, 1). */
nn_real nn_rng_uniform(nn_rng *rng);

/* Uniform double in [lo, hi). */
nn_real nn_rng_uniform_range(nn_rng *rng, nn_real lo, nn_real hi);

/* Standard normal (mean 0, stddev 1), via Box-Muller. */
nn_real nn_rng_normal(nn_rng *rng);

/* --------------------------------------------------------------------- */
/* Activations                                                            */
/* --------------------------------------------------------------------- */

typedef enum nn_activation {
    NN_ACT_LINEAR = 0,
    NN_ACT_SIGMOID,
    NN_ACT_TANH,
    NN_ACT_RELU,
    /*
     * Softmax normalizes a whole layer's outputs together (each output
     * depends on every pre-activation value in the layer, not just its
     * own), so it cannot be expressed as a per-scalar function. nn_forward()
     * special-cases it on the full output vector of a layer; the scalar
     * helpers below treat it as the identity, since it is not meaningful
     * in isolation.
     */
    NN_ACT_SOFTMAX
} nn_activation;

/* Apply an activation function to a pre-activation value. */
nn_real nn_activation_apply(nn_activation act, nn_real x);

/* Derivative of an activation function at a pre-activation value. */
nn_real nn_activation_derivative(nn_activation act, nn_real x);

/* --------------------------------------------------------------------- */
/* Network structure                                                     */
/* --------------------------------------------------------------------- */

/*
 * Weight initialization strategy for nn_create().
 *   NN_INIT_UNIFORM: plain uniform in [-1, 1].
 *   NN_INIT_XAVIER:  Glorot uniform, suited to sigmoid/tanh layers.
 *   NN_INIT_HE:      He uniform, suited to ReLU layers.
 * Biases are always initialized to 0.
 */
typedef enum nn_init {
    NN_INIT_UNIFORM = 0,
    NN_INIT_XAVIER,
    NN_INIT_HE
} nn_init;

/* One fully-connected layer: y = activation(W * x + b). */
typedef struct nn_layer {
    size_t n_inputs;
    size_t n_outputs;
    nn_activation activation;
    nn_real *weights; /* n_outputs * n_inputs, row-major: weights[o * n_inputs + i] */
    nn_real *biases;  /* n_outputs */
} nn_layer;

/* A feed-forward network: an ordered list of layers. */
typedef struct nn_network {
    size_t n_layers;
    nn_layer *layers;
} nn_network;

/*
 * Create a network from an architecture description.
 *
 *   layer_sizes:    e.g. {2, 8, 8, 1} -> 2 inputs, two hidden layers of 8, 1 output.
 *   n_layer_sizes:  length of layer_sizes (>= 2).
 *   activations:    one activation per layer (excludes the input "layer"),
 *                    so its length must be n_layer_sizes - 1.
 *   n_activations:  length of activations, must equal n_layer_sizes - 1.
 *   init:           weight initialization strategy.
 *   seed:           RNG seed, for reproducible initialization.
 *   out:            receives the newly allocated network on NN_OK.
 */
nn_status nn_create(const size_t *layer_sizes, size_t n_layer_sizes,
                     const nn_activation *activations, size_t n_activations,
                     nn_init init, uint64_t seed, nn_network **out);

/* Release a network. nn_free(NULL) is a no-op. */
void nn_free(nn_network *net);

/* Deep copy a network (weights, biases, architecture included). */
nn_status nn_clone(const nn_network *net, nn_network **out);

/* Save a network to a text file. */
nn_status nn_save(const nn_network *net, const char *path);

/* Load a network previously written by nn_save(). */
nn_status nn_load(const char *path, nn_network **out);

/* --------------------------------------------------------------------- */
/* Forward propagation                                                    */
/* --------------------------------------------------------------------- */

/*
 * Reusable scratch space for nn_forward(): one buffer per layer, sized to
 * that layer's output, allocated once and reused across calls so that
 * nn_forward() itself never allocates.
 */
typedef struct nn_forward_buffer {
    size_t n_layers;
    nn_real **activations; /* activations[l] has length net->layers[l].n_outputs */
} nn_forward_buffer;

/* Allocate scratch space sized for net. Must be recreated if net's architecture changes. */
nn_status nn_forward_buffer_create(const nn_network *net, nn_forward_buffer **out);

/* Release scratch space. nn_forward_buffer_free(NULL) is a no-op. */
void nn_forward_buffer_free(nn_forward_buffer *buf);

/*
 * Run a forward pass: output = activation(... activation(W1 * input + b1) ...).
 *
 *   input:  net->layers[0].n_inputs values.
 *   output: net->layers[n_layers - 1].n_outputs values, written on NN_OK.
 *   buf:    scratch space from nn_forward_buffer_create(net, ...); reused
 *           across calls, never (re)allocated here.
 */
nn_status nn_forward(const nn_network *net, nn_forward_buffer *buf,
                      const nn_real *input, nn_real *output);

/* --------------------------------------------------------------------- */
/* Dataset                                                                */
/* --------------------------------------------------------------------- */

/* A labeled dataset: n_samples pairs of (input, target) vectors. */
typedef struct nn_dataset {
    size_t n_samples;
    size_t n_inputs;
    size_t n_outputs;
    nn_real *inputs;  /* n_samples * n_inputs,  row-major: inputs[s * n_inputs + i] */
    nn_real *targets; /* n_samples * n_outputs, row-major: targets[s * n_outputs + o] */
} nn_dataset;

/* Allocate an uninitialized dataset of the given shape. */
nn_status nn_dataset_create(size_t n_samples, size_t n_inputs, size_t n_outputs, nn_dataset **out);

/* Release a dataset. nn_dataset_free(NULL) is a no-op. */
void nn_dataset_free(nn_dataset *ds);

/*
 * Load a dataset from a CSV file: one sample per line, n_inputs values
 * followed by n_outputs values, comma-separated. Blank lines are skipped.
 */
nn_status nn_dataset_load_csv(const char *path, size_t n_inputs, size_t n_outputs, nn_dataset **out);

/* --------------------------------------------------------------------- */
/* Loss                                                                   */
/* --------------------------------------------------------------------- */

/*
 * NN_LOSS_MSE:           mean squared error; pairs with any output
 *                        activation except NN_ACT_SOFTMAX.
 * NN_LOSS_CROSS_ENTROPY: categorical cross-entropy for one-hot targets;
 *                        requires an NN_ACT_SOFTMAX output layer.
 */
typedef enum nn_loss {
    NN_LOSS_MSE = 0,
    NN_LOSS_CROSS_ENTROPY
} nn_loss;

/* Loss between a network output and its target, both of length n. */
nn_real nn_loss_value(nn_loss loss, const nn_real *output, const nn_real *target, size_t n);

/* --------------------------------------------------------------------- */
/* Backpropagation                                                       */
/* --------------------------------------------------------------------- */

/*
 * Gradient of the loss w.r.t. every weight and bias, one array per layer.
 * n_inputs/n_outputs are captured from the network at creation time so the
 * gradient is self-describing; they must stay in sync with the network
 * passed to nn_gradient_apply().
 */
typedef struct nn_gradient {
    size_t n_layers;
    size_t *n_inputs;    /* n_inputs[l]  == net->layers[l].n_inputs  at creation */
    size_t *n_outputs;   /* n_outputs[l] == net->layers[l].n_outputs at creation */
    nn_real **dweights;  /* dweights[l] has length n_inputs[l] * n_outputs[l] */
    nn_real **dbiases;   /* dbiases[l]  has length n_outputs[l] */
} nn_gradient;

nn_status nn_gradient_create(const nn_network *net, nn_gradient **out);
void nn_gradient_free(nn_gradient *grad);

/* Reset all accumulated gradients to 0, e.g. before a new mini-batch. */
void nn_gradient_zero(nn_gradient *grad);

/*
 * net->layers[l].weights[w] -= learning_rate * grad->dweights[l][w] / n_samples
 * (and similarly for biases). n_samples is normally the number of examples
 * accumulated into grad since the last nn_gradient_zero().
 */
void nn_gradient_apply(nn_network *net, const nn_gradient *grad, nn_real learning_rate, size_t n_samples);

/* Reusable scratch space for nn_backprop(): pre/post-activations and deltas, one array per layer. */
typedef struct nn_backprop_buffer {
    size_t n_layers;
    nn_real **z;     /* pre-activation sums,  per layer, length n_outputs */
    nn_real **a;     /* post-activation values, per layer, length n_outputs */
    nn_real **delta; /* dL/dz scratch, per layer, length n_outputs */
} nn_backprop_buffer;

nn_status nn_backprop_buffer_create(const nn_network *net, nn_backprop_buffer **out);
void nn_backprop_buffer_free(nn_backprop_buffer *buf);

/*
 * Run a forward + backward pass for one (input, target) example and ADD
 * its gradient contribution into grad (call nn_gradient_zero() first to
 * start a fresh accumulation, e.g. once per mini-batch).
 *
 * loss must be compatible with the network's output activation, see
 * nn_loss above; an incompatible pairing returns NN_ERR_INVALID_ARG.
 */
nn_status nn_backprop(const nn_network *net, nn_backprop_buffer *buf,
                       const nn_real *input, const nn_real *target,
                       nn_loss loss, nn_gradient *grad);

/* --------------------------------------------------------------------- */
/* Supervised training                                                    */
/* --------------------------------------------------------------------- */

typedef struct nn_train_params {
    nn_real learning_rate;
    size_t epochs;
    size_t batch_size;    /* 0 means "the whole dataset" (batch gradient descent) */
    nn_loss loss;
    uint64_t shuffle_seed; /* reproducible dataset shuffling between epochs */
} nn_train_params;

/*
 * Train net on dataset via mini-batch gradient descent (SGD when
 * batch_size == 1). dataset's dimensions must match net's input/output
 * sizes. The dataset order is reshuffled at the start of every epoch.
 */
nn_status nn_train_supervised(nn_network *net, const nn_dataset *dataset,
                               const nn_train_params *params);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */
