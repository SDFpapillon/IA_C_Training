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

- [x] `nn_forward(net, buf, input, output)` (buf = reusable scratch space, see below)
- [x] Intermediate activation buffers (reused, no malloc per call)
- [x] Softmax for the output layer (classification)
- [x] Tests: known results on a small network with fixed weights

## Phase 3 — Supervised training (backpropagation)

- [x] `nn_dataset` structure (inputs + expected outputs) and loading (simple CSV)
- [x] Loss functions: MSE, cross-entropy
- [x] Gradient backpropagation
- [x] Gradient descent: SGD, mini-batch
- [x] Hyperparameters in a params struct: learning rate, epochs, batch size
- [x] Shuffle the dataset between epochs
- [x] Gradient check via finite differences (test)
- [x] API: `nn_train_supervised(net, dataset, &params)`
- [ ] Optional: momentum, Adam

## Phase 4 — Genetic algorithm training

- [x] Genome ↔ network weights mapping (flatten / rebuild)
- [x] Population: creation, release
- [x] Population seeding from an existing network (clone + mutate) — needed for hybrid training
- [x] User-provided fitness function (callback `nn_real (*fitness)(const nn_network *, void *ctx)`)
- [x] Selection: tournament, roulette
- [x] Crossover: uniform, one-point
- [x] Mutation: Gaussian noise, configurable rate
- [x] Elitism (keep the N best)
- [x] Generation loop + stopping criteria
- [x] API: `nn_train_genetic(population, fitness_cb, ctx, &params)`
- [x] Optional `pre_eval_hook` in the params: called on each individual before evaluation (enables memetic training)

## Phase 5 — Hybrid training (supervised + genetic)

Goal: inject known answers (e.g. "mate in 1 is always the best move") into genetic training,
so the population does not have to rediscover them by chance.

- [x] **Strategy 1 — Fitness bonus**: fitness = task score + bonus × correct answers on known cases.
      No library change needed; provide a helper to score a network on a dataset.
- [x] **Strategy 2 — Pre-train then evolve**: supervised training on known cases, then seed the
      genetic population from that network.
- [x] **Strategy 3 — Memetic**: each individual runs a few gradient steps on known cases before
      evaluation, via `pre_eval_hook`.
- [x] Compare the three strategies against pure genetic on the same task (convergence speed, final score)
      — see `examples/hybrid_training_demo.c` (a comparison tool to run and read, not a pass/fail test).
- [ ] Validate on a small game first (tic-tac-toe, Connect 4) before anything like chess
      — deferred to Phase 9's `examples/tictactoe_hybrid.c`, where real game logic belongs.

## Phase 6 — Visualization

Generalizable metrics (any network):
- [x] Training callback every X iterations / generations (`on_step(net, iteration, ctx)` for
      supervised training, `on_generation(population, generation, ctx)` for genetic — the
      population's cached fitness is what "best/average fitness" needs, a single network isn't enough)
- [x] Export loss and accuracy history (CSV) — `nn_history_logger` + `nn_history_log_step`
- [x] Genetic: export best and average fitness per generation (CSV) — `nn_fitness_logger` +
      `nn_fitness_log_generation`

Specific views:
- [x] Decision boundary map for 2-input classifiers: export prediction grid per snapshot
- [x] Write frames directly as PPM/BMP images (trivial formats, no dependency) — PPM (P6) only;
      BMP adds nothing PPM doesn't already cover for this use case, so skipped to avoid duplicating
      the same "write a trivial image format" work twice
- [x] More than 2 inputs: 2D slice (pick 2 inputs, fix the others) — `nn_export_decision_boundary_ppm`
      takes axis_x/axis_y plus a fixed-inputs vector, so the plain 2-input case is just axis_x=0, axis_y=1
- [x] Regression: predicted vs expected values — `nn_export_predictions_csv`

Viewer:
- [x] Separate HTML viewer (in `tools/`) reading the exported files, based on the existing
      decision-boundary prototype (architecture, learning rate, frames every X iterations, playback speed)
      — no such prototype was found in this repo; built `tools/viewer.html` from scratch instead
      (confirmed with Loïs). Single self-contained file, no server/build step/dependency: loads the
      history CSV, run-metadata JSON and PPM frames as local files, parses PPM/CSV in plain JS, and
      renders a decision-boundary canvas plus a loss/accuracy chart with play/pause, speed and
      frame-scrubbing controls. See `examples/visualize_training.c` for a ready-made export to feed it.
- [x] **Extra, requested after the fact**: build a network and a dataset from `tools/viewer.html` itself
      (a network-architecture editor exporting a small config format, a JSON/CSV dataset editor exporting
      a dataset.csv), train with `examples/train_from_files.c` on the command line (the browser can't run
      C), then load the resulting trained network back into the viewer for a *live* decision-boundary
      preview — a JS re-implementation of `nn_forward()`, checked to match the C output bit-for-bit.
      Deliberately not a "train" button in the browser: that needs either duplicating the training
      engine in JS or compiling the library to WebAssembly, both bigger asks than "edit config, run
      offline, inspect results" (see conversation).

## Phase 7 — Tests

- [x] Home-made mini test framework (`ASSERT` macros, no dependency)
- [x] Unit tests per module — every `src/*.c` has a matching `tests/test_*.c`
      (`nn_fitness_log.c` is covered inside `test_history.c`, alongside `nn_history.c`)
- [x] Integration test: learn XOR (supervised AND genetic) — supervised in `test_train.c`
      (already existed), genetic added in `tests/test_integration.c`
- [x] Integration test: two spirals classification — `tests/test_integration.c`, a small
      one-turn/30-points-per-class version so it trains in ~1s and stays reliable across seeds
      (empirically 93-98% accuracy across seeds 1-5; asserts >= 0.9)
- [x] Memory leak checks (Valgrind on Linux / ASan) — ASan/UBSan already run after every phase;
      this pass additionally ran `valgrind --leak-check=full --errors-for-leak-kinds=all` across
      all 19 test binaries and the example programs: zero leaks, zero errors

## Phase 8 — Windows portability

- [ ] Build with MinGW (`make`) on Windows
- [ ] Optional: MSVC support (CMake or `.bat` script)
- [ ] Avoid non-portable extensions (VLAs, POSIX functions)
- [ ] GitHub Actions CI: Linux (GCC + Clang) and Windows

## Phase 9 — Examples and documentation

- [x] `examples/xor_supervised.c`
- [x] `examples/xor_genetic.c`
- [x] `examples/spirals.c` (with decision boundary export) — honest about the known limitation:
      a plain small MLP without momentum fits the sparse training points well without necessarily
      recovering the true spiral curve between them; that gap *is* why this benchmark is famous,
      not a bug in the example
- [x] `examples/tictactoe_hybrid.c` (hybrid training demo) — game logic + minimax solver live in the
      example (library stays game-agnostic); known cases = minimax-labeled immediate-win/forced-block
      positions; fitness = self-play record against perfect minimax play. Pure genetic happened to
      reach perfect play (0 losses/20) while the Strategy 2 hybrid did worse in this run (10
      losses/20) — a genuine result, not cherry-picked, consistent with Phase 5's finding that hybrid
      strategies don't automatically win
- [x] Fill in the *Building* and *Usage* sections of the README — Building already existed; added a
      minimal runnable Usage snippet plus a table pointing at every example
- [x] Document the API (comments in `nn.h`, possibly Doxygen) — audited and filled remaining gaps
      (e.g. `nn_rng_seed` had none); kept plain comments over Doxygen, consistent with "no external
      dependencies": the project is small enough that generated-docs tooling isn't earning its keep yet
