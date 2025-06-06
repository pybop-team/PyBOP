# [Unreleased](https://github.com/pybop-team/PyBOP)

## Features

- [#644](https://github.com/pybop-team/PyBOP/pull/644) - Adds example applications for common battery experiments.

## Optimisations

## Bug Fixes

- [#737](https://github.com/pybop-team/PyBOP/pull/737) - Sensitivities no longer available for CasadiSolver in Pybamm v25.6 onwards.
- [#705](https://github.com/pybop-team/PyBOP/pull/705) - Bug fix `fitting_problem.evaulate()` failure return type alongside fixes for Pybamm `v25.4`.
- [#546](https://github.com/pybop-team/PyBOP/pull/546) - Default Pybamm solver to `IDAKLU`, changes required for Pybamm v25.4.1

## Breaking Changes

# [v25.3](https://github.com/pybop-team/PyBOP/tree/v25.3) - 2025-03-28

## Features

- [#649](https://github.com/pybop-team/PyBOP/pull/649) - Adds verbose outputs to Pints-based optimisers
- [#659](https://github.com/pybop-team/PyBOP/pull/659) - Enables user-defined weightings of the error measures.
- [#674](https://github.com/pybop-team/PyBOP/issues/674) - Adds the reason for stopping to the `OptimisationResult`.
- [#663](https://github.com/pybop-team/PyBOP/pull/663) - Adds DFN fitting examples alongside synthetic data generation methods.

## Optimisations

- [#676](https://github.com/pybop-team/PyBOP/pull/676) - Update the format of the problem sensitivities to a dict.
- [#681](https://github.com/pybop-team/PyBOP/pull/681) - Update the spatial variable defaults of the `GroupedSPMe` model.
- [#692](https://github.com/pybop-team/PyBOP/pull/692) - Improvements/fixes for `BaseSampler` and `BasePintsSampler` classes, adds `ChainProcessor` classes w/ clearer structure.

## Bug Fixes

- [#678](https://github.com/pybop-team/PyBOP/pull/678) - Fixed bug where model wasn't plotted for observer classes with `pybop.plot.quick()`.

## Breaking Changes

- [#684](https://github.com/pybop-team/PyBOP/pull/684) - Updates `plot.quick` to `plot.problem` for clarity.
- [#661](https://github.com/pybop-team/PyBOP/pull/661) - Adds `pybop.CostInterface` which aligns the optimisers and samplers with a unified `call_cost` in which transformations and sign inversions are applied. Also includes bug fixes for transformations and gradient calculations.

# [v25.1](https://github.com/pybop-team/PyBOP/tree/v25.1) - 2025-02-03

## Features

- [#636](https://github.com/pybop-team/PyBOP/pull/636) - Adds `pybop.IRPropPlus` optimiser with corresponding tests.
- [#635](https://github.com/pybop-team/PyBOP/pull/635) - Adds support for multi-proposal evaluation of list-like objects to `BaseCost` classes.
- [#635](https://github.com/pybop-team/PyBOP/pull/635) - Adds global parameter sensitivity analysis with method `BaseCost.sensitivity_analysis`. This is computation is added to `OptimisationResult` if optimiser arg `compute_sensitivities` is `True`. An additional arg is added to select the number of samples for analysis: `n_sensitivity_samples`.
- [#630] (https://github.com/pybop-team/PyBOP/pull/632) - Fisher Information Matrix added to `BaseLikelihood` class.
- [#619](https://github.com/pybop-team/PyBOP/pull/619) - Adds `pybop.SimulatingAnnealing` optimiser with corresponding tests.
- [#565](https://github.com/pybop-team/PyBOP/pull/627) - DigiBatt added as funding partner.

## Optimisations

- [#638](https://github.com/pybop-team/PyBOP/pull/638) - Allows the problem class to accept any domain name.
- [#618](https://github.com/pybop-team/PyBOP/pull/618) - Adds Mean Absolute Error (MAE) and Mean Squared Error (MSE) costs.
- [#601](https://github.com/pybop-team/PyBOP/pull/601) - Deprecates `MultiOptimisationResult` by merging with `OptimisationResult`.
- [#600](https://github.com/pybop-team/PyBOP/pull/600) - Removes repetitious functionality within the cost classes.

## Bug Fixes

- [#602](https://github.com/pybop-team/PyBOP/pull/602) - Aligns the standard quick plot of `MultiFittingProblem` outputs.

## Breaking Changes

- [#656](https://github.com/pybop-team/PyBOP/pull/656) - Completes `ParameterSet` changes from #593 and aligns the simulation options in `model.predict` with the model properties such as the solver.
- [#593](https://github.com/pybop-team/PyBOP/pull/593) - Enables `ParameterSet` to systematically return a `pybamm.ParameterValues` object within the model class.

# [v24.12](https://github.com/pybop-team/PyBOP/tree/v24.12) - 2024-12-21

## Features

- [#481](https://github.com/pybop-team/PyBOP/pull/481) - Adds experimental support for PyBaMM's jaxified IDAKLU solver. Includes Jax-specific cost functions `pybop.JaxSumSquareError` and `pybop.JaxLogNormalLikelihood`. Adds `Jax` optional dependency to PyBaMM dependency.
- [#597](https://github.com/pybop-team/PyBOP/pull/597) - Adds number of function evaluations `n_evaluations` to `OptimisationResult`.
- [#362](https://github.com/pybop-team/PyBOP/issues/362) - Adds the `classify_using_Hessian` functionality to classify the optimised result.
- [#584](https://github.com/pybop-team/PyBOP/pull/584) - Adds the `GroupedSPMe` model for parameter identification.
- [#571](https://github.com/pybop-team/PyBOP/pull/571) - Adds Multistart functionality to optimisers via initialisation arg `multistart`.
- [#582](https://github.com/pybop-team/PyBOP/pull/582) - Fixes `population_size` arg for Pints' based optimisers, reshapes `parameters.rvs` to be parameter instances.
- [#570](https://github.com/pybop-team/PyBOP/pull/570) - Updates the contour and surface plots, adds mixed chain effective sample size computation, x0 to optim.log
- [#566](https://github.com/pybop-team/PyBOP/pull/566) - Adds `UnitHyperCube` transformation class, fixes incorrect application of gradient transformation.
- [#569](https://github.com/pybop-team/PyBOP/pull/569) - Adds parameter specific learning rate functionality to GradientDescent optimiser.
- [#282](https://github.com/pybop-team/PyBOP/issues/282) - Restructures the examples directory.
- [#396](https://github.com/pybop-team/PyBOP/issues/396) - Adds `ecm_with_tau.py` example script.
- [#452](https://github.com/pybop-team/PyBOP/issues/452) - Extends `cell_mass` and `approximate_capacity` for half-cell models.
- [#544](https://github.com/pybop-team/PyBOP/issues/544) - Allows iterative plotting using `StandardPlot`.
- [#541](https://github.com/pybop-team/PyBOP/pull/541) - Adds `ScaledLogLikelihood` and `BaseMetaLikelihood` classes.
- [#409](https://github.com/pybop-team/PyBOP/pull/409) - Adds plotting and convergence methods for Monte Carlo sampling. Includes open-access Tesla 4680 dataset for Bayesian inference example. Fixes transformations for sampling.
- [#531](https://github.com/pybop-team/PyBOP/pull/531) - Adds Voronoi optimiser surface plot (`pybop.plot.surface`) for fast optimiser aligned cost visualisation.
- [#532](https://github.com/pybop-team/PyBOP/issues/532) - Adds `linked_parameters` example script which shows how to update linked parameters during design optimisation.
- [#529](https://github.com/pybop-team/PyBOP/issues/529) - Adds `GravimetricPowerDensity` and `VolumetricPowerDensity` costs, along with the mathjax extension for Sphinx.

## Optimisations

- [#580](https://github.com/pybop-team/PyBOP/pull/580) - Random Search optimiser is implimented.
- [#588](https://github.com/pybop-team/PyBOP/pull/588) - Makes `minimising` a property of `BaseOptimiser` set by the cost class.
- [#512](https://github.com/pybop-team/PyBOP/pull/513) - Refactors `LogPosterior` with attributes pointing to composed likelihood object.
- [#551](https://github.com/pybop-team/PyBOP/pull/551) - Refactors Optimiser arguments, `population_size` and `max_iterations` as default args, improves optimiser docstrings

## Bug Fixes

- [#595](https://github.com/pybop-team/PyBOP/pull/595) - Fixes non-finite LogTransformed bounds for indices of zero.
- [#561](https://github.com/pybop-team/PyBOP/pull/561) - Bug fixes the sign of the SciPy cost logs for maximised costs.
- [#505](https://github.com/pybop-team/PyBOP/pull/505) - Bug fixes for `LogPosterior` with transformed `GaussianLogLikelihood` likelihood.

## Breaking Changes

- [#481](https://github.com/pybop-team/PyBOP/pull/481) - `problem.model` is now a copied instance of `model`
- [#598](https://github.com/pybop-team/PyBOP/pull/598) - Depreciated `Adam` optimiser has been removed, see `AdamW` for replacement.
- [#531](https://github.com/pybop-team/PyBOP/pull/531) - Plot methods moved to `pybop.plot` with mostly minimal renaming. For example, `pybop.plot_parameters` is now `pybop.plot.parameters`. Other breaking changes include: `pybop.plot2d` to `pybop.plot.contour`.
- [#526](https://github.com/pybop-team/PyBOP/pull/526) - Refactor `OptimisationResults` classes, with `optim.run()` now return the full object. Adds finite cost value check for optimised parameters.

# [v24.9.1](https://github.com/pybop-team/PyBOP/tree/v24.9.1) - 2024-09-16

## Features

## Bug Fixes

- [#495](https://github.com/pybop-team/PyBOP/pull/495) - Bugfixes for Transformation class, adds `apply_transform` optional arg to `BaseCost` for transformation functionality.

## Breaking Changes

# [v24.9.0](https://github.com/pybop-team/PyBOP/tree/v24.9.0) - 2024-09-10

## Features

- [#462](https://github.com/pybop-team/PyBOP/pull/462) - Enables multidimensional learning rate for `pybop.AdamW` with updated (more robust) integration testing. Fixes bug in `Minkowski` and `SumofPower` cost functions for gradient-based optimisers.
- [#411](https://github.com/pybop-team/PyBOP/pull/411) - Updates notebooks with README in `examples/` directory, removes kaleido dependency and moves to nbviewer rendering, displays notebook figures with `notebook_connected` plotly renderer
- [#6](https://github.com/pybop-team/PyBOP/issues/6) - Adds Monte Carlo functionality, with methods based on Pints' algorithms. A base class is added `BaseSampler`, in addition to `PintsBaseSampler`.
- [#353](https://github.com/pybop-team/PyBOP/issues/353) - Allow user-defined check_params functions to enforce nonlinear constraints, and enable SciPy constrained optimisation methods
- [#222](https://github.com/pybop-team/PyBOP/issues/222) - Adds an example for performing and electrode balancing.
- [#441](https://github.com/pybop-team/PyBOP/issues/441) - Adds an example for estimating constants within a `pybamm.FunctionalParameter`.
- [#405](https://github.com/pybop-team/PyBOP/pull/405) - Adds frequency-domain based EIS prediction methods via `model.simulateEIS` and updates to `problem.evaluate` with examples and tests.
- [#460](https://github.com/pybop-team/PyBOP/pull/460) - Notebook example files added for ECM and folder structure updated.
- [#450](https://github.com/pybop-team/PyBOP/pull/450) - Adds support for IDAKLU with output variables, and corresponding examples, tests.
- [#364](https://github.com/pybop-team/PyBOP/pull/364) - Adds the MultiFittingProblem class and the multi_fitting example script.
- [#444](https://github.com/pybop-team/PyBOP/issues/444) - Merge `BaseModel` `build()` and `rebuild()` functionality.
- [#435](https://github.com/pybop-team/PyBOP/pull/435) - Adds SLF001 linting for private members.
- [#418](https://github.com/pybop-team/PyBOP/issues/418) - Wraps the `get_parameter_info` method from PyBaMM to get a dictionary of parameter names and types.
- [#413](https://github.com/pybop-team/PyBOP/pull/413) - Adds `DesignCost` functionality to `WeightedCost` class with additional tests.
- [#357](https://github.com/pybop-team/PyBOP/pull/357) - Adds `Transformation()` class with `LogTransformation()`, `IdentityTransformation()`, and `ScaledTransformation()`, `ComposedTransformation()` implementations with corresponding examples and tests.
- [#427](https://github.com/pybop-team/PyBOP/issues/427) - Adds the nbstripout pre-commit hook to remove unnecessary metadata from notebooks.
- [#327](https://github.com/pybop-team/PyBOP/issues/327) - Adds the `WeightedCost` subclass, defines when to evaluate a problem and adds the `spm_weighted_cost` example script.
- [#393](https://github.com/pybop-team/PyBOP/pull/383) - Adds Minkowski and SumofPower cost classes, with an example and corresponding tests.
- [#403](https://github.com/pybop-team/PyBOP/pull/403/) - Adds lychee link checking action.

## Bug Fixes

- [#473](https://github.com/pybop-team/PyBOP/pull/473) - Bugfixes for transformation class, adds optional `apply_transform` arg to `BaseCost.__call__()`, adds `log_update()` method to `BaseOptimiser`
- [#464](https://github.com/pybop-team/PyBOP/issues/464) - Fix order of design `parameter_set` updates and refactor `update_capacity`.
- [#468](https://github.com/pybop-team/PyBOP/issue/468) - Renames `quick_plot.py` to `standard_plots.py`.
- [#454](https://github.com/pybop-team/PyBOP/issue/454) - Fixes benchmarking suite.
- [#421](https://github.com/pybop-team/PyBOP/issues/421) - Adds a default value for the initial SOC for design problems.

## Breaking Changes

- [#499](https://github.com/pybop-team/PyBOP/pull/499) - BPX is added as an optional dependency.
- [#483](https://github.com/pybop-team/PyBOP/pull/483) - Replaces `pybop.MAP` with `pybop.LogPosterior` with an updated call args and bugfixes.
- [#436](https://github.com/pybop-team/PyBOP/pull/436) - **API Change:** The functionality from `BaseCost.evaluate/S1` & `BaseCost._evaluate/S1` is represented in `BaseCost.__call__` & `BaseCost.compute`. `BaseCost.compute` directly acts on the predictions, while `BaseCost.__call__` calls `BaseProblem.evaluate/S1` before `BaseCost.compute`. `compute` has optional args for gradient cost calculations.
- [#424](https://github.com/pybop-team/PyBOP/issues/424) - Replaces the `init_soc` input to `FittingProblem` with the option to pass an initial OCV value, updates `BaseModel` and fixes `multi_model_identification.ipynb` and `spm_electrode_design.ipynb`.

# [v24.6.1](https://github.com/pybop-team/PyBOP/tree/v24.6.1) - 2024-07-31

## Features

- [#313](https://github.com/pybop-team/PyBOP/pull/313/) - Fixes for PyBaMM v24.5, drops support for PyBaMM v23.9, v24.1

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
