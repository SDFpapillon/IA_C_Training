# Ia_c_training

> A lightweight C library for building and training neural networks — supporting both supervised training (known outputs) and training through genetic algorithms.

## Status

🚧 **Early development** — this project is just getting started. Expect breaking changes, missing features, and an evolving API.

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

Coming soon — the API is not yet defined. Once the core is in place, this section will include a minimal example showing how to:

- Create a neural network
- Train it with a labeled dataset
- Train it using a genetic algorithm

## Roadmap

- [ ] Core network structure (layers, neurons, weights)
- [ ] Forward propagation
- [ ] Supervised training (backpropagation)
- [ ] Genetic algorithm training
- [ ] Windows build support
- [ ] Examples and documentation

## License

MIT — see [LICENSE](LICENSE).

## Author

Personal project by Loïs.
