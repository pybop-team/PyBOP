# [Unreleased](https://github.com/pybop-team/PyBOP)

## Features

- [#249](https://github.com/pybop-team/PyBOP/pull/249) - Add WeppnerHuggins model and GITT example.
- [#304](https://github.com/pybop-team/PyBOP/pull/304) - Decreases the testing suite completion time.
- [#301](https://github.com/pybop-team/PyBOP/pull/301) - Updates default echem solver to "fast with events" mode.
- [#251](https://github.com/pybop-team/PyBOP/pull/251) - Increment PyBaMM > v23.5, remove redundant tests within integration tests, increment citation version, fix examples with incorrect model definitions.
- [#285](https://github.com/pybop-team/PyBOP/pull/285) - Drop support for Python 3.8.
- [#275](https://github.com/pybop-team/PyBOP/pull/275) - Adds Maximum a Posteriori (MAP) cost function with corresponding tests.
- [#273](https://github.com/pybop-team/PyBOP/pull/273) - Adds notebooks to nox examples session and updates CI workflows for change.
- [#250](https://github.com/pybop-team/PyBOP/pull/250) - Adds DFN, MPM, MSMR models and moves multiple construction variables to BaseEChem. Adds exception catch on simulate & simulateS1.
- [#241](https://github.com/pybop-team/PyBOP/pull/241) - Adds experimental circuit model fitting notebook with LG M50 data.
- [#268](https://github.com/pybop-team/PyBOP/pull/268) - Fixes the GitHub Release artifact uploads, allowing verification of
codesigned binaries and source distributions via `sigstore-python`.
- [#79](https://github.com/pybop-team/PyBOP/issues/79) - Adds BPX as a dependency and imports BPX support from PyBaMM.
- [#267](https://github.com/pybop-team/PyBOP/pull/267) - Add classifiers to pyproject.toml, update project.urls.
- [#195](https://github.com/pybop-team/PyBOP/issues/195) - Adds the Nelder-Mead optimiser from PINTS as another option.

## Bug Fixes

- [#330](https://github.com/pybop-team/PyBOP/issues/330) - Fixes implementation of default plotting options.
- [#317](https://github.com/pybop-team/PyBOP/pull/317) - Installs seed packages into `nox` sessions, ensuring that scheduled tests can pass.
- [#308](https://github.com/pybop-team/PyBOP/pull/308) - Enables testing on both macOS Intel and macOS ARM (Silicon) runners and fixes the scheduled tests.
- [#299](https://github.com/pybop-team/PyBOP/pull/299) - Bugfix multiprocessing support for Linux, MacOS, Windows (WSL) and improves coverage.
- [#270](https://github.com/pybop-team/PyBOP/pull/270) - Updates PR template.
- [#91](https://github.com/pybop-team/PyBOP/issues/91) - Adds a check on the number of parameters for CMAES and makes XNES the default optimiser.

# [v24.3](https://github.com/pybop-team/PyBOP/tree/v24.3) - 2024-03-25

## Features

- [#245](https://github.com/pybop-team/PyBOP/pull/245) - Updates ruff config for import linting.
- [#198](https://github.com/pybop-team/PyBOP/pull/198) - Adds default subplot trace options, removes `[]` in axis plots as per SI standard, add varying signal length to quick_plot, restores design optimisation execption.
- [#224](https://github.com/pybop-team/PyBOP/pull/224) - Updated prediction objects to dictionaries, cost class calculations, added `additional_variables` argument to problem class, updated scipy.minimize defualt method to Nelder-Mead, added gradient cost landscape plots with optional argument.
- [#179](https://github.com/pybop-team/PyBOP/pull/203) - Adds `asv` configuration for benchmarking and initial benchmark suite.
- [#218](https://github.com/pybop-team/PyBOP/pull/218) - Adds likelihood base class, `GaussianLogLikelihoodKnownSigma`, `GaussianLogLikelihood`, and `ProbabilityBased` cost function. As well as addition of a maximum likelihood estimation (MLE) example.
- [#185](https://github.com/pybop-team/PyBOP/pull/185) - Adds a pull request template, additional nox sessions `quick` for standard tests + docs, `pre-commit` for pre-commit, `test` to run all standard tests, `doctest` for docs.
- [#215](https://github.com/pybop-team/PyBOP/pull/215) - Adds `release_workflow.md` and updates `release_action.yaml`
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

- [#259](https://github.com/pybop-team/PyBOP/pull/259) - Fix gradient calculation from `model.simulateS1` to remove cross-polution and refactor cost._evaluateS1 for fitting costs.
- [#233](https://github.com/pybop-team/PyBOP/pull/233) - Enforces model rebuild on initialisation of a Problem to allow a change of experiment, fixes if statement triggering current function update, updates `predictions` to `simulation` to keep distinction between `predict` and `simulate` and adds `test_changes`.
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
