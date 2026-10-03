/*
 * Two interleaved spirals, a classic hard benchmark for small MLPs: trains
 * a {2,32,32,1} network to tell the two apart, then exports a decision
 * boundary snapshot (spirals.ppm) and the training points themselves
 * (spirals_dataset.csv, via nn_export_predictions_csv) so you can compare
 * the model's boundary against where the real points land. Open the PPM
 * with any viewer that supports Netpbm (GIMP, `feh`, VS Code's image
 * preview, ...), or load it into tools/viewer.html as a single frame.
 *
 * Kept to one turn and 30 points per class so it trains in about a
 * second; see TODO.md Phase 7 / tests/test_integration.c for the same
 * generator used as a regression test.
 *
 * Don't expect spirals.ppm to look like a clean, tightly-curled spiral:
 * with plain SGD and no momentum (Phase 3's "Optional: momentum, Adam"
 * isn't implemented), a network this size typically settles for a simpler
 * boundary that still gets ~97% of the sparse training points right
 * without actually tracing the curve between them. That gap between
 * "high accuracy on the points" and "learned the true shape" is exactly
 * why two-spirals is a known hard benchmark; it's not a bug here. Turning
 * up TURNS, point count, hidden width and epochs pushes toward a truer
 * spiral, at the cost of (much) longer training.
 */
#include <math.h>
#include <stdio.h>

#include "nn.h"

#define SPIRALS_PI 3.14159265358979323846
#define N_PER_CLASS 30
#define TURNS 1.0

static void build_two_spirals(nn_dataset **out)
{
    nn_dataset_create(N_PER_CLASS * 2, 2, 1, out);
    nn_dataset *ds = *out;
    for (size_t i = 0; i < N_PER_CLASS; i++) {
        nn_real t = (nn_real)i / (nn_real)(N_PER_CLASS - 1);
        nn_real angle = t * TURNS * 2.0 * SPIRALS_PI;
        nn_real radius = t;

        ds->inputs[i * 2 + 0] = radius * cos((double)angle);
        ds->inputs[i * 2 + 1] = radius * sin((double)angle);
        ds->targets[i] = 0.0;

        size_t j = N_PER_CLASS + i;
        ds->inputs[j * 2 + 0] = radius * cos((double)angle + SPIRALS_PI);
        ds->inputs[j * 2 + 1] = radius * sin((double)angle + SPIRALS_PI);
        ds->targets[j] = 1.0;
    }
}

int main(void)
{
    nn_dataset *ds = NULL;
    build_two_spirals(&ds);

    size_t sizes[] = {2, 32, 32, 1};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_TANH, NN_ACT_SIGMOID};
    nn_network *net = NULL;
    nn_create(sizes, 4, acts, 3, NN_INIT_XAVIER, /*seed=*/1, &net);

    nn_train_params params = {0};
    params.learning_rate = 0.3;
    params.epochs = 5000;
    params.batch_size = 0;
    params.loss = NN_LOSS_MSE;
    params.shuffle_seed = 1;
    nn_train_supervised(net, ds, &params);

    nn_forward_buffer *buf = NULL;
    nn_forward_buffer_create(net, &buf);

    nn_real scratch[1];
    nn_real accuracy = 0.0;
    nn_dataset_accuracy(net, buf, scratch, ds, &accuracy);
    printf("accuracy on the training points: %.1f%%\n", (double)accuracy * 100.0);

    nn_real fixed_inputs[2] = {0.0, 0.0}; /* both axes are swept, so unused */
    nn_export_decision_boundary_ppm(net, buf, fixed_inputs, 0, 1, -1.2, 1.2, -1.2, 1.2, 128, 128,
                                     "spirals.ppm");
    nn_export_predictions_csv(net, buf, ds, "spirals_dataset.csv");
    printf("wrote spirals.ppm (decision boundary) and spirals_dataset.csv (points + predictions)\n");

    nn_forward_buffer_free(buf);
    nn_dataset_free(ds);
    nn_free(net);
    return 0;
}
