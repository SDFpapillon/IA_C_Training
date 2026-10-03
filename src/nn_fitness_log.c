#include <stdio.h>
#include <stdlib.h>

#include "nn.h"

struct nn_fitness_logger {
    FILE *file;
};

nn_status nn_fitness_logger_create(const char *path, nn_fitness_logger **out)
{
    if (path == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }

    nn_fitness_logger *logger = malloc(sizeof(*logger));
    if (logger == NULL) {
        return NN_ERR_ALLOC;
    }
    logger->file = fopen(path, "w");
    if (logger->file == NULL) {
        free(logger);
        return NN_ERR_IO;
    }

    fprintf(logger->file, "generation,best_fitness,avg_fitness\n");
    *out = logger;
    return NN_OK;
}

void nn_fitness_logger_free(nn_fitness_logger *logger)
{
    if (logger == NULL) {
        return;
    }
    if (logger->file != NULL) {
        fclose(logger->file);
    }
    free(logger);
}

void nn_fitness_log_generation(const nn_population *pop, size_t generation, void *ctx)
{
    nn_fitness_logger *logger = ctx;

    nn_real best = pop->fitness[0];
    nn_real sum = pop->fitness[0];
    for (size_t i = 1; i < pop->size; i++) {
        if (pop->fitness[i] > best) {
            best = pop->fitness[i];
        }
        sum += pop->fitness[i];
    }
    nn_real avg = sum / (nn_real)pop->size;

    fprintf(logger->file, "%zu,%.17g,%.17g\n", generation, (double)best, (double)avg);
}
