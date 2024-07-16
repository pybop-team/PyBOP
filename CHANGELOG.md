# [Unreleased](https://github.com/pybop-team/PyBOP)

## Features


- [#6](https://github.com/pybop-team/PyBOP/issues/6) - Adds Monte Carlo functionality, with methods based on Pints' algorithms. A base class is added `BaseSampler`, in addition to `PintsBaseSampler`.
- [#393](https://github.com/pybop-team/PyBOP/pull/383) - Adds Minkowski and SumofPower cost classes, with an example and corresponding tests.
- [#403](https://github.com/pybop-team/PyBOP/pull/403/) - Adds lychee link checking action.

## Bug Fixes


## Breaking Changes

# [v24.6](https://github.com/pybop-team/PyBOP/tree/v24.6) - 2024-07-08

## Features

- [#319](https://github.com/pybop-team/PyBOP/pull/319/) - Adds `CuckooSearch` optimiser with corresponding tests.
- [#359](https://github.com/pybop-team/PyBOP/pull/359/) - Aligning Inputs between problem, observer and model.
- [#379](https://github.com/pybop-team/PyBOP/pull/379) - Adds model.simulateS1 to weekly benchmarks.
- [#174](https://github.com/pybop-team/PyBOP/issues/174) - Adds new logo and updates Readme for accessibility.
- [#316](https://github.com/pybop-team/PyBOP/pull/316) - Adds Adam with weight decay (AdamW) optimiser, adds depreciation warning for pints.Adam implementation.
- [#271](https://github.com/pybop-team/PyBOP/issues/271) - Aligns the output of the optimisers via a generalisation of Result class.
- [#315](https://github.com/pybop-team/PyBOP/pull/315) - Updates __init__ structure to remove circular import issues and minimises dependancy imports across codebase for faster PyBOP module import. Adds type-hints to BaseModel and refactors rebuild parameter variables.
- [#236](https://github.com/pybop-team/PyBOP/issues/236) - Restructures the optimiser classes, adds a new optimisation API through direct construction and keyword arguments, and fixes the setting of `max_iterations`, and `_minimising`. Introduces `pybop.BaseOptimiser`, `pybop.BasePintsOptimiser`, and `pybop.BaseSciPyOptimiser` classes.
- [#322](https://github.com/pybop-team/PyBOP/pull/322) - Add `Parameters` class to store and access multiple parameters in one object.
- [#321](https://github.com/pybop-team/PyBOP/pull/321) - Updates Prior classes with BaseClass, adds a `problem.sample_initial_conditions` method to improve stability of SciPy.Minimize optimiser.
- [#249](https://github.com/pybop-team/PyBOP/pull/249) - Add WeppnerHuggins model and GITT example.
- [#304](https://github.com/pybop-team/PyBOP/pull/304) - Decreases the testing suite completion time.
- [#301](https://github.com/pybop-team/PyBOP/pull/301) - Updates default echem solver to "fast with events" mode.
- [#251](https://github.com/pybop-team/PyBOP/pull/251) - Increment PyBaMM > v23.5, remove redundant tests within integration tests, increment citation version, fix examples with incorrect model definitions.
- [#285](https://github.com/pybop-team/PyBOP/pull/285) - Drop support for Python 3.8.
- [#275](https://github.com/pybop-team/PyBOP/pull/275) - Adds Maximum a Posteriori (MAP) cost function with corresponding tests.
- [#273](https://github.com/pybop-team/PyBOP/pull/273) - Adds notebooks to nox examples session and updates CI workflows for change.
- [#250](https://github.com/pybop-team/PyBOP/pull/250) - Adds DFN, MPM, MSMR models and moves multiple construction variables to BaseEChem. Adds exception catch on simulate & simulateS1.
- [#241](https://github.com/pybop-team/PyBOP/pull/241) - Adds experimental circuit model fitting notebook with LG M50 data.
- [#268](https://github.com/pybop-team/PyBOP/pull/268) - Fixes the GitHub Release artifact uploads, allowing verification of codesigned binaries and source distributions via `sigstore-python`.
- [#79](https://github.com/pybop-team/PyBOP/issues/79) - Adds BPX as a dependency and imports BPX support from PyBaMM.
- [#267](https://github.com/pybop-team/PyBOP/pull/267) - Add classifiers to pyproject.toml, update project.urls.
- [#195](https://github.com/pybop-team/PyBOP/issues/195) - Adds the Nelder-Mead optimiser from PINTS as another option.

## Bug Fixes

- [#393](https://github.com/pybop-team/PyBOP/pull/393) - General integration test fixes. Adds UserWarning when using Plot2d with prior generated bounds.
- [#338](https://github.com/pybop-team/PyBOP/pull/338) - Fixes GaussianLogLikelihood class, adds integration tests, updates non-bounded parameter implementation by applying bounds from priors and `boundary_multiplier` argument. Bugfixes to CMAES construction.
- [#339](https://github.com/pybop-team/PyBOP/issues/339) - Updates the calculation of the cyclable lithium capacity in the spme_max_energy example.
- [#387](https://github.com/pybop-team/PyBOP/issues/387) - Adds keys to ParameterSet and updates ECM OCV check.
- [#380](https://github.com/pybop-team/PyBOP/pull/380) - Restore self._boundaries construction for `pybop.PSO`
- [#372](https://github.com/pybop-team/PyBOP/pull/372) - Converts `np.array` to `np.asarray` for Numpy v2.0 support.
- [#165](https://github.com/pybop-team/PyBOP/issues/165) - Stores the attempted and best parameter values and the best cost for each iteration in the log attribute of the optimiser and updates the associated plots.
- [#354](https://github.com/pybop-team/PyBOP/issues/354) - Fixes the calculation of the gradient in the `RootMeanSquaredError` cost.
- [#347](https://github.com/pybop-team/PyBOP/issues/347) - Resets options between MSMR tests to cope with a bug in PyBaMM v23.9 which is fixed in PyBaMM v24.1.
- [#337](https://github.com/pybop-team/PyBOP/issues/337) - Restores benchmarks, relaxes CI schedule for benchmarks and scheduled tests.
- [#231](https://github.com/pybop-team/PyBOP/issues/231) - Allows passing of keyword arguments to PyBaMM models and disables build on initialisation.
- [#321](https://github.com/pybop-team/PyBOP/pull/321) - Improves `integration/test_spm_parameterisation.py` stability, adds flakly pytest plugin, and `test_thevenin_parameterisation.py` integration test.
- [#330](https://github.com/pybop-team/PyBOP/issues/330) - Fixes implementation of default plotting options.
- [#317](https://github.com/pybop-team/PyBOP/pull/317) - Installs seed packages into `nox` sessions, ensuring that scheduled tests can pass.
- [#308](https://github.com/pybop-team/PyBOP/pull/308) - Enables testing on both macOS Intel and macOS ARM (Silicon) runners and fixes the scheduled tests.
- [#299](https://github.com/pybop-team/PyBOP/pull/299) - Bugfix multiprocessing support for Linux, MacOS, Windows (WSL) and improves coverage.
- [#270](https://github.com/pybop-team/PyBOP/pull/270) - Updates PR template.
- [#91](https://github.com/pybop-team/PyBOP/issues/91) - Adds a check on the number of parameters for CMAES and makes XNES the default optimiser.

## Breaking Changes

- [#322](https://github.com/pybop-team/PyBOP/pull/322) - Add `Parameters` class to store and access multiple parameters in one object (API change).
- [#285](https://github.com/pybop-team/PyBOP/pull/285) - Drop support for Python 3.8.
- [#251](https://github.com/pybop-team/PyBOP/pull/251) - Drop support for PyBaMM v23.5
- [#236](https://github.com/pybop-team/PyBOP/issues/236) - Restructures the optimiser classes (API change).

# [v24.3.1](https://github.com/pybop-team/PyBOP/tree/v24.3.1) - 2024-06-17

## Features


## Bug Fixes

- [#369](https://github.com/pybop-team/PyBOP/pull/369) - Upper pins Numpy < 2.0 due to breaking Pints' functionality.

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
