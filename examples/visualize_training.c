/*
 * Trains a small XOR classifier and exports the data Phase 6's viewer is
 * meant to read: a loss/accuracy history CSV, and a handful of decision
 * boundary PPM snapshots taken every few epochs. Run it from the repo
 * root; it writes xor_history.csv and xor_frame_NNNN.ppm to the current
 * directory.
 */
#include <stdio.h>

#include "nn.h"

typedef struct {
    nn_history_logger *logger;
    nn_forward_buffer *frame_buf;
    size_t frame_every;
} viz_ctx;

static void on_step(const nn_network *net, size_t iteration, void *ctx_)
{
    viz_ctx *ctx = ctx_;
    nn_history_log_step(net, iteration, ctx->logger);

    if (iteration % ctx->frame_every == 0) {
        char path[64];
        snprintf(path, sizeof(path), "xor_frame_%04zu.ppm", iteration);
        nn_real fixed_inputs[2] = {0.0, 0.0};
        nn_export_decision_boundary_ppm(net, ctx->frame_buf, fixed_inputs, 0, 1, -0.5, 1.5, -0.5, 1.5,
                                         64, 64, path);
    }
}

int main(void)
{
    size_t sizes[] = {2, 8, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    if (nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 1, &net) != NN_OK) {
        fprintf(stderr, "nn_create failed\n");
        return 1;
    }

    nn_dataset *ds = NULL;
    if (nn_dataset_create(4, 2, 1, &ds) != NN_OK) {
        fprintf(stderr, "nn_dataset_create failed\n");
        return 1;
    }
    nn_real inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    nn_real targets[] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; i++) {
        ds->inputs[i] = inputs[i];
    }
    for (size_t i = 0; i < 4; i++) {
        ds->targets[i] = targets[i];
    }

    viz_ctx ctx;
    ctx.frame_every = 20;
    if (nn_history_logger_create("xor_history.csv", net, ds, NN_LOSS_MSE, &ctx.logger) != NN_OK) {
        fprintf(stderr, "nn_history_logger_create failed\n");
        return 1;
    }
    if (nn_forward_buffer_create(net, &ctx.frame_buf) != NN_OK) {
        fprintf(stderr, "nn_forward_buffer_create failed\n");
        return 1;
    }

    nn_train_params params = {0};
    params.learning_rate = 0.5;
    params.epochs = 200;
    params.batch_size = 0;
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 1;
    params.on_step = on_step;
    params.on_step_ctx = &ctx;
    params.on_step_every = 1;

    if (nn_train_supervised(net, ds, &params) != NN_OK) {
        fprintf(stderr, "nn_train_supervised failed\n");
        return 1;
    }

    /* A small sidecar the HTML viewer (tools/viewer.html) reads for display
     * only; the library itself has no notion of "a training run". */
    FILE *meta = fopen("xor_meta.json", "w");
    if (meta != NULL) {
        fprintf(meta,
                "{\"architecture\":[%zu,%zu,%zu],\"learning_rate\":%.17g,"
                "\"epochs\":%zu,\"frame_every\":%zu}\n",
                sizes[0], sizes[1], sizes[2], (double)params.learning_rate, params.epochs,
                ctx.frame_every);
        fclose(meta);
    }

    printf("Wrote xor_history.csv, xor_meta.json and xor_frame_*.ppm (every %zu epochs) to the "
           "current directory.\n",
           ctx.frame_every);

    nn_forward_buffer_free(ctx.frame_buf);
    nn_history_logger_free(ctx.logger);
    nn_dataset_free(ds);
    nn_free(net);
    return 0;
}
