# [Unreleased](https://github.com/pybop-team/PyBOP)

## Features

- [#151](https://github.com/pybop-team/PyBOP/issues/151) - Adds a standalone version of the Problem class.

## Bug Fixes

- [#164](https://github.com/pybop-team/PyBOP/issues/164) - Fixes convergence issues with gradient-based optimisers, changes default `model.check_params()` to allow infeasible solutions during optimisation iterations. Adds a feasibility check on the optimal parameters.

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

# [v23.11](https://github.com/pybop-team/PyBOP/releases/tag/v23.11)
- Initial release
- Adds Pints, NLOpt, and SciPy optimisers
- Adds SumofSquareError and RootMeanSquareError cost functions
- Adds Parameter and Dataset classes
