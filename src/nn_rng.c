#include <math.h>

#include "nn.h"

#define NN_PI 3.14159265358979323846

void nn_rng_seed(nn_rng *rng, uint64_t seed)
{
    rng->state = seed;
}

uint64_t nn_rng_next_u64(nn_rng *rng)
{
    /* splitmix64: fast, deterministic, good enough for weight init and
     * mutation. Not cryptographically secure. */
    uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

nn_real nn_rng_uniform(nn_rng *rng)
{
    /* Top 53 bits -> double in [0, 1), full mantissa precision. */
    uint64_t bits = nn_rng_next_u64(rng) >> 11;
    return (nn_real)bits * (1.0 / 9007199254740992.0); /* 1 / 2^53 */
}

nn_real nn_rng_uniform_range(nn_rng *rng, nn_real lo, nn_real hi)
{
    return lo + (hi - lo) * nn_rng_uniform(rng);
}

nn_real nn_rng_normal(nn_rng *rng)
{
    nn_real u1 = nn_rng_uniform(rng);
    nn_real u2 = nn_rng_uniform(rng);
    if (u1 < 1e-300) {
        u1 = 1e-300; /* avoid log(0) */
    }
    return sqrt(-2.0 * log((double)u1)) * cos(2.0 * NN_PI * (double)u2);
}
