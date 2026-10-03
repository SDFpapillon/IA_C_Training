#include <stdlib.h>
#include <string.h>

#include "nn.h"

void nn_mutate(nn_real *genome, size_t n, nn_real rate, nn_real stddev, nn_rng *rng)
{
    for (size_t i = 0; i < n; i++) {
        if (nn_rng_uniform(rng) < rate) {
            genome[i] += nn_rng_normal(rng) * stddev;
        }
    }
}

void nn_crossover_apply(const nn_real *parent_a, const nn_real *parent_b, nn_real *child,
                         size_t n, nn_crossover method, nn_rng *rng)
{
    if (method == NN_CROSSOVER_ONE_POINT) {
        size_t point = (size_t)(nn_rng_next_u64(rng) % n);
        for (size_t i = 0; i < n; i++) {
            child[i] = (i < point) ? parent_a[i] : parent_b[i];
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            child[i] = (nn_rng_uniform(rng) < 0.5) ? parent_a[i] : parent_b[i];
        }
    }
}

static size_t nn_select_tournament(const nn_real *fitness, size_t n, size_t k, nn_rng *rng)
{
    size_t best = (size_t)(nn_rng_next_u64(rng) % n);
    for (size_t t = 1; t < k; t++) {
        size_t candidate = (size_t)(nn_rng_next_u64(rng) % n);
        if (fitness[candidate] > fitness[best]) {
            best = candidate;
        }
    }
    return best;
}

static size_t nn_select_roulette(const nn_real *fitness, size_t n, nn_real total_fitness, nn_rng *rng)
{
    if (total_fitness <= 0.0) {
        return (size_t)(nn_rng_next_u64(rng) % n);
    }
    nn_real r = nn_rng_uniform(rng) * total_fitness;
    nn_real cumulative = 0.0;
    for (size_t i = 0; i < n; i++) {
        cumulative += fitness[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n - 1; /* floating-point rounding fallback */
}

size_t nn_select(const nn_real *fitness, size_t n, nn_selection method,
                  size_t tournament_size, nn_real total_fitness, nn_rng *rng)
{
    if (method == NN_SELECT_ROULETTE) {
        return nn_select_roulette(fitness, n, total_fitness, rng);
    }
    size_t k = tournament_size < n ? tournament_size : n;
    return nn_select_tournament(fitness, n, k, rng);
}

typedef struct nn_rank_entry {
    size_t index;
    nn_real fitness;
} nn_rank_entry;

static int nn_rank_entry_cmp_desc(const void *pa, const void *pb)
{
    const nn_rank_entry *a = pa;
    const nn_rank_entry *b = pb;
    if (a->fitness > b->fitness) {
        return -1;
    }
    if (a->fitness < b->fitness) {
        return 1;
    }
    return 0;
}

nn_status nn_train_genetic(nn_population *pop, nn_fitness_fn fitness, void *fitness_ctx,
                            const nn_genetic_params *params)
{
    if (pop == NULL || fitness == NULL || params == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (params->elitism > pop->size) {
        return NN_ERR_INVALID_ARG;
    }

    size_t n = pop->size;
    size_t genome_len = pop->genome_length;

    nn_real *parent_genomes = malloc(n * genome_len * sizeof(*parent_genomes));
    nn_real *child_genomes = malloc(n * genome_len * sizeof(*child_genomes));
    nn_rank_entry *ranking = malloc(n * sizeof(*ranking));
    if (parent_genomes == NULL || child_genomes == NULL || ranking == NULL) {
        free(parent_genomes);
        free(child_genomes);
        free(ranking);
        return NN_ERR_ALLOC;
    }

    nn_rng rng;
    nn_rng_seed(&rng, params->seed);

    for (size_t gen = 0; gen < params->generations; gen++) {
        if (params->pre_eval_hook != NULL) {
            for (size_t i = 0; i < n; i++) {
                params->pre_eval_hook(pop->networks[i], params->pre_eval_ctx);
            }
        }
        for (size_t i = 0; i < n; i++) {
            pop->fitness[i] = fitness(pop->networks[i], fitness_ctx);
        }

        if (params->on_generation != NULL) {
            size_t every = params->on_generation_every == 0 ? 1 : params->on_generation_every;
            if (gen % every == 0) {
                params->on_generation(pop, gen, params->on_generation_ctx);
            }
        }

        nn_real best_fitness = pop->fitness[0];
        for (size_t i = 1; i < n; i++) {
            if (pop->fitness[i] > best_fitness) {
                best_fitness = pop->fitness[i];
            }
        }
        if (params->has_target_fitness && best_fitness >= params->target_fitness) {
            break;
        }
        if (gen + 1 == params->generations) {
            break; /* last generation already evaluated; no need to reproduce further */
        }

        /* Snapshot this generation's genomes and rank them before any mutation. */
        nn_real total_fitness = 0.0;
        for (size_t i = 0; i < n; i++) {
            nn_genome_flatten(pop->networks[i], &parent_genomes[i * genome_len]);
            ranking[i].index = i;
            ranking[i].fitness = pop->fitness[i];
            total_fitness += pop->fitness[i];
        }
        qsort(ranking, n, sizeof(*ranking), nn_rank_entry_cmp_desc);

        /* Elitism: carry the best individuals over unchanged. */
        for (size_t i = 0; i < params->elitism; i++) {
            size_t src = ranking[i].index;
            memcpy(&child_genomes[i * genome_len], &parent_genomes[src * genome_len],
                   genome_len * sizeof(*child_genomes));
        }

        /* Offspring: selection + crossover + mutation. */
        for (size_t i = params->elitism; i < n; i++) {
            size_t pa = nn_select(pop->fitness, n, params->selection, params->tournament_size,
                                   total_fitness, &rng);
            size_t pb = nn_select(pop->fitness, n, params->selection, params->tournament_size,
                                   total_fitness, &rng);
            nn_real *child = &child_genomes[i * genome_len];
            nn_crossover_apply(&parent_genomes[pa * genome_len], &parent_genomes[pb * genome_len],
                                child, genome_len, params->crossover, &rng);
            nn_mutate(child, genome_len, params->mutation_rate, params->mutation_stddev, &rng);
        }

        /* Install the next generation. */
        for (size_t i = 0; i < n; i++) {
            nn_genome_unflatten(pop->networks[i], &child_genomes[i * genome_len]);
        }
    }

    free(parent_genomes);
    free(child_genomes);
    free(ranking);
    return NN_OK;
}
