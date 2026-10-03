#include <stdio.h>
#include <stdlib.h>

#include "nn.h"

struct nn_history_logger {
    FILE *file;
    nn_forward_buffer *buf;
    nn_real *scratch;
    const nn_dataset *dataset;
    nn_loss loss;
};

nn_status nn_history_logger_create(const char *path, const nn_network *net, const nn_dataset *dataset,
                                    nn_loss loss, nn_history_logger **out)
{
    if (path == NULL || net == NULL || dataset == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_history_logger *logger = malloc(sizeof(*logger));
    if (logger == NULL) {
        return NN_ERR_ALLOC;
    }
    logger->dataset = dataset;
    logger->loss = loss;
    logger->file = NULL;
    logger->buf = NULL;
    logger->scratch = NULL;

    logger->file = fopen(path, "w");
    nn_status status = nn_forward_buffer_create(net, &logger->buf);
    size_t n_out = net->layers[net->n_layers - 1].n_outputs;
    logger->scratch = malloc(n_out * sizeof(*logger->scratch));

    if (logger->file == NULL || status != NN_OK || logger->scratch == NULL) {
        nn_status fail_status = logger->file == NULL ? NN_ERR_IO
                                 : status != NN_OK    ? status
                                                       : NN_ERR_ALLOC;
        nn_history_logger_free(logger);
        return fail_status;
    }

    fprintf(logger->file, "iteration,loss,accuracy\n");
    *out = logger;
    return NN_OK;
}

void nn_history_logger_free(nn_history_logger *logger)
{
    if (logger == NULL) {
        return;
    }
    if (logger->file != NULL) {
        fclose(logger->file);
    }
    nn_forward_buffer_free(logger->buf);
    free(logger->scratch);
    free(logger);
}

void nn_history_log_step(const nn_network *net, size_t iteration, void *ctx)
{
    nn_history_logger *logger = ctx;
    nn_real loss = 0.0;
    nn_real accuracy = 0.0;
    nn_dataset_loss(net, logger->buf, logger->scratch, logger->dataset, logger->loss, &loss);
    nn_dataset_accuracy(net, logger->buf, logger->scratch, logger->dataset, &accuracy);
    fprintf(logger->file, "%zu,%.17g,%.17g\n", iteration, (double)loss, (double)accuracy);
}
