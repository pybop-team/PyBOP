# [Unreleased](https://github.com/pybop-team/PyBOP)

## Features

- [#204](https://github.com/pybop-team/PyBOP/pull/204) - Splits integration, unit, examples, plots tests, update workflows. Adds pytest `--examples`, `--integration`, `--plots` args. Adds tests for coverage after removal of examples. Adds examples and integrations nox sessions. Adds `pybop.RMSE._evaluateS1()` method
- [#206](https://github.com/pybop-team/PyBOP/pull/206) - Adds Python 3.12 support with corresponding github actions changes.
- [#18](https://github.com/pybop-team/PyBOP/pull/18) - Adds geometric parameter fitting capability, via `model.rebuild()` with `model.rebuild_parameters`.
- [#203](https://github.com/pybop-team/PyBOP/pull/203) - Adds support for modern Python packaging via a `pyproject.toml` file and configures the `pytest` test runner and `ruff` linter to use their configurations stored as declarative metadata.
- [#123](https://github.com/pybop-team/PyBOP/issues/123) - Configures scheduled tests to run against the last three PyPI releases of PyBaMM via dynamic GitHub Actions matrix generation.
- [#187](https://github.com/pybop-team/PyBOP/issues/187) - Adds M1 Github runner to `test_on_push` workflow, updt. self-hosted supported python versions in scheduled tests.
- [#118](https://github.com/pybop-team/PyBOP/issues/118) - Adds example jupyter notebooks.
- [#151](https://github.com/pybop-team/PyBOP/issues/151) - Adds a standalone version of the Problem class.
- [#12](https://github.com/pybop-team/PyBOP/issues/12) - Adds initial implementation of an Observer class and an unscented Kalman filter.
- [#190](https://github.com/pybop-team/PyBOP/issues/190) - Adds a second example design cost, namely the VolumetricEnergyDensity.

## Bug Fixes

- [#123](https://github.com/pybop-team/PyBOP/issues/123) - Reinstates check for availability of parameter sets via PyBaMM upon retrieval by `pybop.ParameterSet.pybamm()`.
- [#196](https://github.com/pybop-team/PyBOP/issues/196) - Fixes failing observer cost tests.
- [#63](https://github.com/pybop-team/PyBOP/issues/63) - Removes NLOpt Optimiser from future releases. This is to support deployment to the Apple M-Series platform.
- [#164](https://github.com/pybop-team/PyBOP/issues/164) - Fixes convergence issues with gradient-based optimisers, changes default `model.check_params()` to allow infeasible solutions during optimisation iterations. Adds a feasibility check on the optimal parameters.
- [#211](https://github.com/pybop-team/PyBOP/issues/211) - Allows a subset of parameter bounds or bounds=None to be passed, returning warnings where needed.

# [v23.12](https://github.com/pybop-team/PyBOP/tree/v23.12) - 2023-12-19

## Features

- [#141](https://github.com/pybop-team/PyBOP/pull/141) - Adds documentation with Sphinx and PyData Sphinx Theme. Updates docstrings across package, relocates `costs` and `dataset` to top-level of package. Adds noxfile session and deployment workflow for docs.
- [#131](https://github.com/pybop-team/PyBOP/issues/131) - Adds `SciPyDifferentialEvolution` optimiser, adds functionality for user-selectable maximum iteration limit to `SciPyMinimize`, `NLoptOptimize`, and `BaseOptimiser` classes.
- [#107](https://github.com/pybop-team/PyBOP/issues/107) - Adds Equivalent Circuit Model (ECM) with examples, Import/Export parameter methods `ParameterSet.import_parameter` and `ParameterSet.export_parameters`, updates default FittingProblem.signal definition to `"Voltage [V]"`, and testing infrastructure
- [#127](https://github.com/pybop-team/PyBOP/issues/127) - Adds Windows and macOS runners to the `test_on_push` action
- [#114](https://github.com/pybop-team/PyBOP/issues/114) - Adds standard plotting class `pybop.StandardPlot()` via plotly backend
- [#114](https://github.com/pybop-team/PyBOP/issues/114) - Adds `quick_plot()`, `plot_convergence()`, and `plot_cost2d()` methods
- [#114](https://github.com/pybop-team/PyBOP/issues/114) - Adds a SciPy minimize example and logging for non-Pints optimisers
- [#116](https://github.com/pybop-team/PyBOP/issues/116) - Adds PSO, SNES, XNES, ADAM, and IPropMin optimisers to PintsOptimisers() class
- [#38](https://github.com/pybop-team/PyBOP/issues/38) - Restructures the Problem classes ahead of adding a design optimisation example
- [#38](https://github.com/pybop-team/PyBOP/issues/38) - Updates tests and adds a design optimisation example script `spme_max_energy`
- [#120](https://github.com/pybop-team/PyBOP/issues/120) - Updates the parameterisation test settings including the number of iterations
- [#145](https://github.com/pybop-team/PyBOP/issues/145) - Reformats Dataset to contain a dictionary and signal into a list of strings

## Bug Fixes

- [#182](https://github.com/pybop-team/PyBOP/pull/182) - Allow square-brackets indexing of Dataset

# [v23.11](https://github.com/pybop-team/PyBOP/releases/tag/v23.11)
- Initial release
- Adds Pints, NLOpt, and SciPy optimisers
- Adds SumofSquareError and RootMeanSquareError cost functions
- Adds Parameter and Dataset classes
