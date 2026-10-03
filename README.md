# Ia_c_training

> A lightweight C library for building and training neural networks — supporting both supervised training (known outputs) and training through genetic algorithms.

## Status

🚧 **Under active development**, but the core is in place and working: network creation, forward
propagation, supervised training (backpropagation), genetic algorithm training, hybrid
supervised+genetic strategies, CSV/PPM export, and an HTML training viewer (`tools/viewer.html`).
Windows build support is the main piece still pending. Expect the API to keep evolving.

## Goals

- Pure C implementation, no external dependencies beyond the standard library
- Support for supervised training (labeled/known-output datasets)
- Support for training via genetic algorithms
- Cross-platform: Linux and Windows

## Requirements

- A C compiler (GCC, Clang, or MSVC/MinGW on Windows)
- `make`

## Building

```bash
make                # static library -> build/release/libiactraining.a
make BUILD=debug    # debug build
make test           # build and run the tests
make examples       # build the examples
make clean
```

On Windows, use MinGW-w64 from [MSYS2](https://www.msys2.org/) (`pacman -S mingw-w64-ucrt-x86_64-gcc make`).

## Usage

```c
#include "nn.h"

/* 2 inputs -> 8 hidden (tanh) -> 1 output (sigmoid). */
size_t sizes[] = {2, 8, 1};
nn_activation acts[] = {NN_ACT_TANH, NN_ACT_SIGMOID};
nn_network *net = NULL;
nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, /*seed=*/1, &net);

/* A labeled dataset: fill ds->inputs / ds->targets yourself, or load a CSV. */
nn_dataset *ds = NULL;
nn_dataset_create(4, 2, 1, &ds);
/* ds->inputs[sample * n_inputs + i], ds->targets[sample * n_outputs + o] ... */

nn_train_params params = {0};
params.learning_rate = 0.5;
params.epochs = 2000;
params.loss = NN_LOSS_MSE;
nn_train_supervised(net, ds, &params);

nn_forward_buffer *buf = NULL;
nn_forward_buffer_create(net, &buf);
nn_real output[1];
nn_forward(net, buf, ds->inputs, output);

nn_save(net, "trained.txt"); /* reload later with nn_load() */

nn_forward_buffer_free(buf);
nn_dataset_free(ds);
nn_free(net);
```

Every `nn_*_create()` call can fail (returns `nn_status`, `NN_OK` on success — checks omitted above
for brevity; see `nn_status_str()` for human-readable error messages).

Training by genetic algorithm instead of backpropagation follows the same shape: build an
`nn_population` with `nn_population_create_random()`, write a fitness function, and call
`nn_train_genetic()`. See the examples below for complete, runnable code for both, plus hybrid
training (supervised + genetic combined) and visualization export.

### Examples

Build them with `make examples`, then run from `build/release/examples/`:

| File | Demonstrates |
|---|---|
| `xor_supervised.c` | The smallest possible supervised-training example |
| `xor_genetic.c` | The same task, solved by evolution instead |
| `spirals.c` | A harder 2D classification task, with decision-boundary image export |
| `visualize_training.c` | Exporting loss/accuracy history + decision-boundary snapshots for `tools/viewer.html` |
| `train_from_files.c` | Training from an architecture config + dataset file (what `tools/viewer.html`'s network/dataset editors export) |
| `hybrid_training_demo.c` | Comparing pure genetic training against the three hybrid strategies from `TODO.md` |
| `tictactoe_hybrid.c` | Hybrid training on a real (tiny) game: known tactical positions + genetic self-play against a minimax opponent |

Open `tools/viewer.html` directly in a browser (no server needed) to inspect a training run, build a
network/dataset interactively, or preview a trained network's decision boundary live.

## Roadmap

- [x] Core network structure (layers, neurons, weights)
- [x] Forward propagation
- [x] Supervised training (backpropagation)
- [x] Genetic algorithm training
- [ ] Windows build support
- [x] Examples and documentation

See `TODO.md` for the detailed, phase-by-phase breakdown (including hybrid training and the
visualization pipeline, both already implemented).

## License

MIT — see [LICENSE](LICENSE).

## Author

Personal project by Loïs.
