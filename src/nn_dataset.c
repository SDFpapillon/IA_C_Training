#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

#define NN_CSV_MAX_LINE 4096

nn_status nn_dataset_create(size_t n_samples, size_t n_inputs, size_t n_outputs, nn_dataset **out)
{
    if (out == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (n_samples == 0 || n_inputs == 0 || n_outputs == 0) {
        return NN_ERR_INVALID_ARG;
    }

    nn_dataset *ds = malloc(sizeof(*ds));
    if (ds == NULL) {
        return NN_ERR_ALLOC;
    }
    ds->n_samples = n_samples;
    ds->n_inputs = n_inputs;
    ds->n_outputs = n_outputs;
    ds->inputs = malloc(n_samples * n_inputs * sizeof(*ds->inputs));
    ds->targets = malloc(n_samples * n_outputs * sizeof(*ds->targets));
    if (ds->inputs == NULL || ds->targets == NULL) {
        nn_dataset_free(ds);
        return NN_ERR_ALLOC;
    }

    *out = ds;
    return NN_OK;
}

void nn_dataset_free(nn_dataset *ds)
{
    if (ds == NULL) {
        return;
    }
    free(ds->inputs);
    free(ds->targets);
    free(ds);
}

static int nn_csv_line_is_blank(const char *line)
{
    for (const char *p = line; *p != '\0'; p++) {
        if (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            return 0;
        }
    }
    return 1;
}

/* Parse exactly n_fields comma-separated numbers from line into values. */
static nn_status nn_csv_parse_line(const char *line, nn_real *values, size_t n_fields)
{
    const char *p = line;

    for (size_t i = 0; i < n_fields; i++) {
        char *end;
        double v = strtod(p, &end);
        if (end == p) {
            return NN_ERR_FORMAT;
        }
        values[i] = (nn_real)v;
        p = end;
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (i + 1 < n_fields) {
            if (*p != ',') {
                return NN_ERR_FORMAT;
            }
            p++;
            while (*p == ' ' || *p == '\t') {
                p++;
            }
        }
    }

    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
        p++;
    }
    return *p == '\0' ? NN_OK : NN_ERR_FORMAT;
}

nn_status nn_dataset_load_csv(const char *path, size_t n_inputs, size_t n_outputs, nn_dataset **out)
{
    if (path == NULL || out == NULL) {
        return NN_ERR_NULL_ARG;
    }
    if (n_inputs == 0 || n_outputs == 0) {
        return NN_ERR_INVALID_ARG;
    }

    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return NN_ERR_IO;
    }

    char line[NN_CSV_MAX_LINE];
    nn_status status = NN_OK;
    size_t n_samples = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        if (len == sizeof(line) - 1 && line[len - 1] != '\n') {
            status = NN_ERR_FORMAT; /* line too long to be a real data row */
            goto done;
        }
        if (!nn_csv_line_is_blank(line)) {
            n_samples++;
        }
    }
    if (n_samples == 0) {
        status = NN_ERR_FORMAT;
        goto done;
    }

    nn_dataset *ds = NULL;
    status = nn_dataset_create(n_samples, n_inputs, n_outputs, &ds);
    if (status != NN_OK) {
        goto done;
    }

    rewind(f);
    size_t n_fields = n_inputs + n_outputs;
    nn_real *row = malloc(n_fields * sizeof(*row));
    if (row == NULL) {
        nn_dataset_free(ds);
        status = NN_ERR_ALLOC;
        goto done;
    }

    size_t s = 0;
    while (s < n_samples && fgets(line, sizeof(line), f) != NULL) {
        if (nn_csv_line_is_blank(line)) {
            continue;
        }
        status = nn_csv_parse_line(line, row, n_fields);
        if (status != NN_OK) {
            break;
        }
        memcpy(&ds->inputs[s * n_inputs], row, n_inputs * sizeof(*row));
        memcpy(&ds->targets[s * n_outputs], &row[n_inputs], n_outputs * sizeof(*row));
        s++;
    }
    free(row);

    if (status != NN_OK) {
        nn_dataset_free(ds);
        goto done;
    }

    *out = ds;

done:
    fclose(f);
    return status;
}
