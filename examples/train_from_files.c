/*
 * Trains a network whose architecture and dataset come from external files
 * instead of being hard-coded, so tools/viewer.html's "build a network" /
 * "build a dataset" editors have something to feed. Usage:
 *
 *   train_from_files <architecture.cfg> <dataset.csv> [output_network.txt] [history.csv]
 *
 * architecture.cfg is a small text format (not a library format, just a
 * convenience for this example):
 *
 *   NNARCH1
 *   <n_layer_sizes>
 *   <size_0> <size_1> ... <size_{n-1}>
 *   <activation_1> ... <activation_{n-1}>   (one per layer, excludes the input)
 *   <init: uniform|xavier|he>
 *   <seed>
 *
 * e.g. for {2, 8, 1} with tanh/sigmoid, Xavier init, seed 1:
 *
 *   NNARCH1
 *   3
 *   2 8 1
 *   tanh sigmoid
 *   xavier
 *   1
 *
 * dataset.csv is the existing nn_dataset_load_csv format (no header: each
 * row is n_inputs values then n_outputs values, comma-separated); the
 * number of input/output columns is inferred from the architecture.
 *
 * Training hyperparameters are fixed constants below rather than CLI flags,
 * since only architecture and dataset were asked to be made configurable.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

#define TRAIN_EPOCHS 2000
#define TRAIN_LEARNING_RATE 0.5
#define HISTORY_EVERY 5

static nn_activation parse_activation(const char *name)
{
    if (strcmp(name, "linear") == 0) return NN_ACT_LINEAR;
    if (strcmp(name, "sigmoid") == 0) return NN_ACT_SIGMOID;
    if (strcmp(name, "tanh") == 0) return NN_ACT_TANH;
    if (strcmp(name, "relu") == 0) return NN_ACT_RELU;
    if (strcmp(name, "softmax") == 0) return NN_ACT_SOFTMAX;
    fprintf(stderr, "unknown activation '%s'\n", name);
    exit(1);
}

static nn_init parse_init(const char *name)
{
    if (strcmp(name, "uniform") == 0) return NN_INIT_UNIFORM;
    if (strcmp(name, "xavier") == 0) return NN_INIT_XAVIER;
    if (strcmp(name, "he") == 0) return NN_INIT_HE;
    fprintf(stderr, "unknown init '%s'\n", name);
    exit(1);
}

/* Reads architecture.cfg and builds the (untrained) network. */
static nn_status load_network_from_config(const char *path, nn_network **out)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return NN_ERR_IO;
    }

    char magic[16] = {0};
    size_t n_sizes = 0;
    if (fscanf(f, "%15s", magic) != 1 || strcmp(magic, "NNARCH1") != 0 ||
        fscanf(f, "%zu", &n_sizes) != 1 || n_sizes < 2) {
        fclose(f);
        return NN_ERR_FORMAT;
    }

    size_t *sizes = malloc(n_sizes * sizeof(*sizes));
    nn_activation *acts = malloc((n_sizes - 1) * sizeof(*acts));
    if (sizes == NULL || acts == NULL) {
        free(sizes);
        free(acts);
        fclose(f);
        return NN_ERR_ALLOC;
    }

    nn_status status = NN_OK;
    for (size_t i = 0; i < n_sizes; i++) {
        if (fscanf(f, "%zu", &sizes[i]) != 1) {
            status = NN_ERR_FORMAT;
            goto done;
        }
    }
    for (size_t i = 0; i < n_sizes - 1; i++) {
        char name[16];
        if (fscanf(f, "%15s", name) != 1) {
            status = NN_ERR_FORMAT;
            goto done;
        }
        acts[i] = parse_activation(name);
    }

    char init_name[16];
    uint64_t seed;
    if (fscanf(f, "%15s", init_name) != 1 || fscanf(f, "%" SCNu64, &seed) != 1) {
        status = NN_ERR_FORMAT;
        goto done;
    }
    nn_init init = parse_init(init_name);

    status = nn_create(sizes, n_sizes, acts, n_sizes - 1, init, seed, out);

done:
    free(sizes);
    free(acts);
    fclose(f);
    return status;
}

static void on_step(const nn_network *net, size_t iteration, void *ctx)
{
    nn_history_log_step(net, iteration, ctx);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <architecture.cfg> <dataset.csv> [output_network.txt] [history.csv]\n",
                argv[0]);
        return 1;
    }
    const char *arch_path = argv[1];
    const char *dataset_path = argv[2];
    const char *output_path = argc > 3 ? argv[3] : "trained_network.txt";
    const char *history_path = argc > 4 ? argv[4] : "training_history.csv";

    nn_network *net = NULL;
    nn_status status = load_network_from_config(arch_path, &net);
    if (status != NN_OK) {
        fprintf(stderr, "failed to load architecture from %s: %s\n", arch_path, nn_status_str(status));
        return 1;
    }

    nn_dataset *ds = NULL;
    status = nn_dataset_load_csv(dataset_path, net->layers[0].n_inputs,
                                  net->layers[net->n_layers - 1].n_outputs, &ds);
    if (status != NN_OK) {
        fprintf(stderr, "failed to load dataset from %s: %s\n", dataset_path, nn_status_str(status));
        return 1;
    }

    /* Cross-entropy requires a softmax output layer and vice versa (see
     * nn_backprop()); infer the only valid choice from the architecture
     * instead of hard-coding one, since the editor lets users pick softmax. */
    nn_loss loss = net->layers[net->n_layers - 1].activation == NN_ACT_SOFTMAX
                       ? NN_LOSS_CROSS_ENTROPY
                       : NN_LOSS_MSE;

    nn_history_logger *logger = NULL;
    status = nn_history_logger_create(history_path, net, ds, loss, &logger);
    if (status != NN_OK) {
        fprintf(stderr, "failed to create history logger: %s\n", nn_status_str(status));
        return 1;
    }

    nn_train_params params = {0};
    params.learning_rate = TRAIN_LEARNING_RATE;
    params.epochs = TRAIN_EPOCHS;
    params.batch_size = 0;
    params.loss = loss;
    params.shuffle_seed = 1;
    params.on_step = on_step;
    params.on_step_ctx = logger;
    params.on_step_every = HISTORY_EVERY;

    status = nn_train_supervised(net, ds, &params);
    if (status != NN_OK) {
        fprintf(stderr, "training failed: %s\n", nn_status_str(status));
        return 1;
    }

    status = nn_save(net, output_path);
    if (status != NN_OK) {
        fprintf(stderr, "failed to save trained network to %s: %s\n", output_path, nn_status_str(status));
        return 1;
    }

    printf("Trained on %zu samples for %zu epochs.\n", ds->n_samples, params.epochs);
    printf("Wrote %s (trained network) and %s (loss/accuracy history).\n", output_path, history_path);

    nn_history_logger_free(logger);
    nn_dataset_free(ds);
    nn_free(net);
    return 0;
}
