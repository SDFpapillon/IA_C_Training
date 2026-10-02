# TODO

Task list for **Ia_c_training**, derived from the README roadmap.
Suggested order: top to bottom, each phase builds on the previous one.

## Design principles

- The network does **not** know how it is trained: training functions take a network and modify it.
  This keeps supervised, genetic and hybrid training composable without special cases.
- Architecture is fully configurable: number of layers, neurons per layer, activation per layer.
- The library stays pure C; it **exports** data, and visualization happens outside of it.

## Phase 0 — Project setup

- [x] Directory layout: `include/`, `src/`, `tests/`, `examples/`, `tools/`
- [x] Base `Makefile` (static lib `libiactraining.a`, targets `all`, `test`, `examples`, `clean`)
- [x] Strict compiler flags: `-std=c99 -Wall -Wextra -Wpedantic` (+ `-g` / `-O2` depending on build)
- [x] `.gitignore` (objects, binaries, `.a`, build directories)
- [x] License: MIT (`LICENSE`)
- [x] Conventions: public symbol prefix (e.g. `nn_`), float type → `nn_real` = `double`
- [x] Error handling strategy (return codes / `nn_status` enum)

## Phase 1 — Network structure

- [x] Public header `include/nn.h`
- [x] Structures: `nn_layer` (weights, biases, activation), `nn_network` (list of layers)
- [x] Architecture described by an array, e.g. `{2, 8, 8, 1}` → 2 inputs, two hidden layers of 8, 1 output
- [x] Activation selectable per layer
- [x] `nn_create(...)` / `nn_free(...)` — clean allocation and release
- [x] Weight initialization (uniform random, Xavier/He) with a controllable seed
- [x] Internal portable, reproducible RNG (do not rely on `rand()`, which differs across platforms)
- [x] Activation functions: sigmoid, tanh, ReLU, linear (+ their derivatives)
- [x] Network copy / clone (needed for genetic training and hybrid seeding)
- [x] Save / load a network (simple binary or text format)

## Phase 2 — Forward propagation

- [ ] `nn_forward(net, input, output)`
- [ ] Intermediate activation buffers (reused, no malloc per call)
- [ ] Softmax for the output layer (classification)
- [ ] Tests: known results on a small network with fixed weights

## Phase 3 — Supervised training (backpropagation)

- [ ] `nn_dataset` structure (inputs + expected outputs) and loading (simple CSV)
- [ ] Loss functions: MSE, cross-entropy
- [ ] Gradient backpropagation
- [ ] Gradient descent: SGD, mini-batch
- [ ] Hyperparameters in a params struct: learning rate, epochs, batch size
- [ ] Shuffle the dataset between epochs
- [ ] Gradient check via finite differences (test)
- [ ] API: `nn_train_supervised(net, dataset, &params)`
- [ ] Optional: momentum, Adam

## Phase 4 — Genetic algorithm training

- [ ] Genome ↔ network weights mapping (flatten / rebuild)
- [ ] Population: creation, release
- [ ] Population seeding from an existing network (clone + mutate) — needed for hybrid training
- [ ] User-provided fitness function (callback `float (*fitness)(nn_network *, void *ctx)`)
- [ ] Selection: tournament, roulette
- [ ] Crossover: uniform, one-point
- [ ] Mutation: Gaussian noise, configurable rate
- [ ] Elitism (keep the N best)
- [ ] Generation loop + stopping criteria
- [ ] API: `nn_train_genetic(population, fitness_cb, ctx, &params)`
- [ ] Optional `pre_eval_hook` in the params: called on each individual before evaluation (enables memetic training)

## Phase 5 — Hybrid training (supervised + genetic)

Goal: inject known answers (e.g. "mate in 1 is always the best move") into genetic training,
so the population does not have to rediscover them by chance.

- [ ] **Strategy 1 — Fitness bonus**: fitness = task score + bonus × correct answers on known cases.
      No library change needed; provide a helper to score a network on a dataset.
- [ ] **Strategy 2 — Pre-train then evolve**: supervised training on known cases, then seed the
      genetic population from that network.
- [ ] **Strategy 3 — Memetic**: each individual runs a few gradient steps on known cases before
      evaluation, via `pre_eval_hook`.
- [ ] Compare the three strategies against pure genetic on the same task (convergence speed, final score)
- [ ] Validate on a small game first (tic-tac-toe, Connect 4) before anything like chess

## Phase 6 — Visualization

Generalizable metrics (any network):
- [ ] Training callback every X iterations / generations (`on_step(net, iteration, ctx)`)
- [ ] Export loss and accuracy history (CSV)
- [ ] Genetic: export best and average fitness per generation (CSV)

Specific views:
- [ ] Decision boundary map for 2-input classifiers: export prediction grid per snapshot
- [ ] Write frames directly as PPM/BMP images (trivial formats, no dependency)
- [ ] More than 2 inputs: 2D slice (pick 2 inputs, fix the others)
- [ ] Regression: predicted vs expected values

Viewer:
- [ ] Separate HTML viewer (in `tools/`) reading the exported files, based on the existing
      decision-boundary prototype (architecture, learning rate, frames every X iterations, playback speed)

## Phase 7 — Tests

- [x] Home-made mini test framework (`ASSERT` macros, no dependency)
- [ ] Unit tests per module
- [ ] Integration test: learn XOR (supervised AND genetic)
- [ ] Integration test: two spirals classification
- [ ] Memory leak checks (Valgrind on Linux / ASan)

## Phase 8 — Windows portability

- [ ] Build with MinGW (`make`) on Windows
- [ ] Optional: MSVC support (CMake or `.bat` script)
- [ ] Avoid non-portable extensions (VLAs, POSIX functions)
- [ ] GitHub Actions CI: Linux (GCC + Clang) and Windows

## Phase 9 — Examples and documentation

- [ ] `examples/xor_supervised.c`
- [ ] `examples/xor_genetic.c`
- [ ] `examples/spirals.c` (with decision boundary export)
- [ ] `examples/tictactoe_hybrid.c` (hybrid training demo)
- [ ] Fill in the *Building* and *Usage* sections of the README
- [ ] Document the API (comments in `nn.h`, possibly Doxygen)
