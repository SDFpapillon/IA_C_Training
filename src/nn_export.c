#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

nn_status nn_export_predictions_csv(const nn_network *net, nn_forward_buffer *buf,
                                     const nn_dataset *dataset, const char *path)
{
    if (net == NULL || buf == NULL || dataset == NULL || path == NULL) {
        return NN_ERR_NULL_ARG;
    }
    size_t last = net->n_layers - 1;
    if (dataset->n_inputs != net->layers[0].n_inputs ||
        dataset->n_outputs != net->layers[last].n_outputs) {
        return NN_ERR_SIZE_MISMATCH;
    }

    FILE *f = fopen(path, "w");
    if (f == NULL) {
        return NN_ERR_IO;
    }

    size_t n_in = dataset->n_inputs;
    size_t n_out = dataset->n_outputs;
    nn_real *output = malloc(n_out * sizeof(*output));
    if (output == NULL) {
        fclose(f);
        return NN_ERR_ALLOC;
    }

    fprintf(f, "sample");
    for (size_t i = 0; i < n_in; i++) {
        fprintf(f, ",input_%zu", i);
    }
    for (size_t o = 0; o < n_out; o++) {
        fprintf(f, ",expected_%zu", o);
    }
    for (size_t o = 0; o < n_out; o++) {
        fprintf(f, ",predicted_%zu", o);
    }
    fprintf(f, "\n");

    nn_status status = NN_OK;
    for (size_t s = 0; s < dataset->n_samples; s++) {
        const nn_real *in = &dataset->inputs[s * n_in];
        const nn_real *target = &dataset->targets[s * n_out];
        status = nn_forward(net, buf, in, output);
        if (status != NN_OK) {
            break;
        }

        fprintf(f, "%zu", s);
        for (size_t i = 0; i < n_in; i++) {
            fprintf(f, ",%.17g", (double)in[i]);
        }
        for (size_t o = 0; o < n_out; o++) {
            fprintf(f, ",%.17g", (double)target[o]);
        }
        for (size_t o = 0; o < n_out; o++) {
            fprintf(f, ",%.17g", (double)output[o]);
        }
        fprintf(f, "\n");
    }

    free(output);
    if (fclose(f) != 0 && status == NN_OK) {
        status = NN_ERR_IO;
    }
    return status;
}

#define NN_PALETTE_SIZE 8
static const unsigned char NN_PALETTE[NN_PALETTE_SIZE][3] = {
    {230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200},
    {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
};

nn_status nn_export_decision_boundary_ppm(const nn_network *net, nn_forward_buffer *buf,
                                           const nn_real *fixed_inputs, size_t axis_x, size_t axis_y,
                                           nn_real x_min, nn_real x_max, nn_real y_min, nn_real y_max,
                                           size_t width, size_t height, const char *path)
{
    if (net == NULL || buf == NULL || fixed_inputs == NULL || path == NULL) {
        return NN_ERR_NULL_ARG;
    }
    size_t n_in = net->layers[0].n_inputs;
    size_t n_out = net->layers[net->n_layers - 1].n_outputs;
    if (axis_x >= n_in || axis_y >= n_in || axis_x == axis_y || width == 0 || height == 0) {
        return NN_ERR_INVALID_ARG;
    }

    nn_real *input = malloc(n_in * sizeof(*input));
    nn_real *output = malloc(n_out * sizeof(*output));
    if (input == NULL || output == NULL) {
        free(input);
        free(output);
        return NN_ERR_ALLOC;
    }
    memcpy(input, fixed_inputs, n_in * sizeof(*input));

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(input);
        free(output);
        return NN_ERR_IO;
    }

    fprintf(f, "P6\n%zu %zu\n255\n", width, height);

    nn_status status = NN_OK;
    for (size_t row = 0; row < height; row++) {
        /* Row 0 is the top of the image, which maps to y_max. */
        nn_real row_frac = (nn_real)row / (nn_real)(height > 1 ? height - 1 : 1);
        input[axis_y] = y_max - (y_max - y_min) * row_frac;

        for (size_t col = 0; col < width; col++) {
            nn_real col_frac = (nn_real)col / (nn_real)(width > 1 ? width - 1 : 1);
            input[axis_x] = x_min + (x_max - x_min) * col_frac;

            status = nn_forward(net, buf, input, output);
            if (status != NN_OK) {
                goto done;
            }

            unsigned char rgb[3];
            if (n_out == 1) {
                nn_real v = output[0];
                if (v < 0.0) {
                    v = 0.0;
                }
                if (v > 1.0) {
                    v = 1.0;
                }
                unsigned char gray = (unsigned char)(v * 255.0);
                rgb[0] = gray;
                rgb[1] = gray;
                rgb[2] = gray;
            } else {
                size_t best = 0;
                for (size_t o = 1; o < n_out; o++) {
                    if (output[o] > output[best]) {
                        best = o;
                    }
                }
                const unsigned char *color = NN_PALETTE[best % NN_PALETTE_SIZE];
                rgb[0] = color[0];
                rgb[1] = color[1];
                rgb[2] = color[2];
            }
            fwrite(rgb, 1, 3, f);
        }
    }

done:
    free(input);
    free(output);
    if (fclose(f) != 0 && status == NN_OK) {
        status = NN_ERR_IO;
    }
    return status;
}
