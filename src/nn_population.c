#include <stdlib.h>

#include "nn.h"

void nn_population_free(nn_population *pop)
{
    if (pop == NULL) {
        return;
    }
    if (pop->networks != NULL) {
        for (size_t i = 0; i < pop->size; i++) {
            nn_free(pop->networks[i]);
        }
    }
    free(pop->networks);
    free(pop->fitness);
    free(pop);
}

nn_status nn_population_create_random(const size_t *layer_sizes, size_t n_layer_sizes,
                                       const nn_activation *activations, size_t n_activations,
                                       nn_init init, size_t pop_size, uint64_t seed,
                                       nn_population **out)
{
    if (layer_sizes == NULL || activations == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (pop_size == 0) {
        return NN_ERR_INVALID_ARG;
    }

    nn_population *pop = malloc(sizeof(*pop));
    if (pop == NULL) {
        return NN_ERR_ALLOC;
    }
    pop->size = pop_size;
    pop->genome_length = 0;
    pop->networks = calloc(pop_size, sizeof(*pop->networks));
    pop->fitness = calloc(pop_size, sizeof(*pop->fitness));
    if (pop->networks == NULL || pop->fitness == NULL) {
        nn_population_free(pop);
        return NN_ERR_ALLOC;
    }

    for (size_t i = 0; i < pop_size; i++) {
        nn_status status = nn_create(layer_sizes, n_layer_sizes, activations, n_activations,
                                      init, seed + (uint64_t)i, &pop->networks[i]);
        if (status != NN_OK) {
            nn_population_free(pop);
            return status;
        }
    }

    pop->genome_length = nn_genome_size(pop->networks[0]);
    *out = pop;
    return NN_OK;
}

nn_status nn_population_create_seeded(const nn_network *seed_net, size_t pop_size,
                                       nn_real mutation_rate, nn_real mutation_stddev,
                                       uint64_t seed, nn_population **out)
{
    if (seed_net == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (pop_size == 0) {
        return NN_ERR_INVALID_ARG;
    }

    nn_population *pop = malloc(sizeof(*pop));
    if (pop == NULL) {
        return NN_ERR_ALLOC;
    }
    pop->size = pop_size;
    pop->genome_length = nn_genome_size(seed_net);
    pop->networks = calloc(pop_size, sizeof(*pop->networks));
    pop->fitness = calloc(pop_size, sizeof(*pop->fitness));
    if (pop->networks == NULL || pop->fitness == NULL) {
        nn_population_free(pop);
        return NN_ERR_ALLOC;
    }

    nn_real *genome = malloc(pop->genome_length * sizeof(*genome));
    if (genome == NULL) {
        nn_population_free(pop);
        return NN_ERR_ALLOC;
    }

    nn_rng rng;
    nn_rng_seed(&rng, seed);

    for (size_t i = 0; i < pop_size; i++) {
        nn_status status = nn_clone(seed_net, &pop->networks[i]);
        if (status != NN_OK) {
            free(genome);
            nn_population_free(pop);
            return status;
        }
        if (i > 0) { /* individual 0 stays an exact, unmutated copy of seed_net */
            nn_genome_flatten(pop->networks[i], genome);
            nn_mutate(genome, pop->genome_length, mutation_rate, mutation_stddev, &rng);
            nn_genome_unflatten(pop->networks[i], genome);
        }
    }

    free(genome);
    *out = pop;
    return NN_OK;
}
