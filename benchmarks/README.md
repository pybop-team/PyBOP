# Benchmarking Directory for PyBOP

Welcome to the benchmarking directory for PyBOP. We use `asv` (airspeed velocity) for benchmarking, which is a tool for running Python benchmarks over time in a consistent environment. This document will guide you through the setup, execution, and viewing of benchmarks.

## Quick Links

- [Airspeed Velocity (asv) Documentation](https://asv.readthedocs.io/)

## Prerequisites

Before you can run benchmarks, you need to ensure that `asv` is installed and that you have a working Python environment. It is also recommended to run benchmarks in a clean, dedicated virtual environment to avoid any side-effects from your local environment.

### Installing `asv`

You can install `asv` using `pip`. It's recommended to do this within a virtual environment:

```bash
pip install asv
```

## Setting Up Benchmarks

The `benchmarks` directory already contains a set of benchmarks for the package. To add or modify benchmarks, edit the `.py` files within this directory.

Each benchmark file should contain one or more classes with methods that `asv` will automatically recognize as benchmarks. Here's an example structure for a benchmark file:

```python
class ExampleBenchmarks:
    def setup(self):
        # Code to run before each benchmark method is executed
        pass

    def time_example_benchmark(self):
        # The actual benchmark code
        pass

    def teardown(self):
        # Code to run after each benchmark method is executed
        pass
```

## Running Benchmarks

With `asv` installed and your benchmarks set up, you can now run benchmarks using the following standard `asv` commands:

### Running All Benchmarks

To run all benchmarks in your python env:

```bash
asv run --python=same
```

This will test the current state of your codebase by default. You can specify a range of commits to run benchmarks against by appending a commit range to the command, like so:

```bash
asv run <commit-hash-1>..<commit-hash-2>
```

### Running Specific Benchmarks

To run a specific benchmark, use:

```bash
asv run --bench <benchmark name>
```

### Running Benchmarks for a Specific Environment

To run benchmarks against a specific Python version:

```bash
asv run --python=same  # To use the same Python version as the current environment
asv run --python=3.8  # To specify the Python version
```

## Viewing Benchmark Results

After running benchmarks, `asv` will generate results which can be viewed as a web page:

```bash
asv publish
asv preview
```

Now you can open your web browser to the URL provided by `asv` to view the results.

## Continuous Benchmarking

You can also set up `asv` for continuous benchmarking where it will track the performance over time. This typically involves integration with a continuous integration (CI) system.

For more detailed instructions on setting up continuous benchmarking, consult the [asv documentation](https://asv.readthedocs.io/en/stable/using.html#continuous-benchmarking).

## Reporting Issues

If you encounter any issues or have suggestions for improving the benchmarks, please open an issue or a pull request in the project repository.

Thank you for contributing to the performance of the package!
