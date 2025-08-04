<div align="center">

  <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/logo/PyBOP_logo_flat.png" alt="logo.svg" width="600" />

  # Python Battery Optimisation and Parameterisation


  [![Scheduled](https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml/badge.svg)](https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml)
  [![Contributors](https://img.shields.io/github/contributors/pybop-team/PyBOP)](https://github.com/pybop-team/PyBOP/graphs/contributors)
  [![Python Versions from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpybop-team%2FPyBOP%2Fdevelop%2Fpyproject.toml&label=Python)](https://pypi.org/project/pybop/)
  [![Codecov](https://codecov.io/gh/pybop-team/PyBOP/branch/develop/graph/badge.svg)](https://codecov.io/gh/pybop-team/PyBOP)
  [![License](https://img.shields.io/github/license/pybop-team/PyBOP?color=blue)](https://github.com/pybop-team/PyBOP/blob/develop/LICENSE)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybop-team/PyBOP/blob/develop/)
  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/pybop-team/PyBOP/tree/develop/examples/notebooks/)
  [![Static Badge](https://img.shields.io/badge/https%3A%2F%2Fpybop-team.github.io%2Fpybop-bench%2F?label=Benchmarks)](https://pybop-team.github.io/pybop-bench/)
  [![Releases](https://img.shields.io/github/v/release/pybop-team/PyBOP?color=gold)](https://github.com/pybop-team/PyBOP/releases)

[<ins>Main Branch Examples</ins>](https://github.com/pybop-team/PyBOP/tree/main/examples) [<ins>Develop Branch Examples</ins>](https://github.com/pybop-team/PyBOP/tree/develop/examples)

</div>

PyBOP provides a complete set of tools for parameterisation and optimisation of battery models, using both Bayesian and frequentist approaches, with [example workflows](https://github.com/pybop-team/PyBOP/tree/main/examples/) to assist the user. PyBOP can be used to parameterise various battery models, including electrochemical and equivalent circuit models available in [PyBaMM](https://pybamm.org/). PyBOP prioritises clear and informative diagnostics for the user, while also allowing for advanced probabilistic methods.

The diagram below shows the conceptual framework of PyBOP. This package is currently under development, so users can expect the API to evolve with future releases.

<p align="center">
    <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/PyBOP-high-level.svg" alt="pybop_arch.svg" width="700" />
</p>

## Installation

Within your virtual environment, install PyBOP:

```bash
pip install pybop
```

To install the most recent state of PyBOP, install from the `develop` branch,

```bash
pip install git+https://github.com/pybop-team/PyBOP.git@develop
```

To install a previous version of PyBOP, use the following template and replace the version number:

```bash
pip install pybop==v24.3
```

To check that PyBOP is installed correctly, run one of the examples in the following section. For a development installation, see the [Contribution Guide](https://github.com/pybop-team/PyBOP/blob/develop/CONTRIBUTING.md#Installation). More installation information is available in our [documentation](https://pybop-docs.readthedocs.io/en/latest/installation.html) and the [extended installation instructions](https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html) for PyBaMM.

## Using PyBOP
PyBOP has two intended uses:

1. Parameter inference from battery test data.

2. Design optimisation under battery manufacturing/use constraints.

These include a wide variety of optimisation problems that require careful consideration due to the choice of battery model, data availability and/or the choice of design parameters.

### Jupyter Notebooks

Explore our [example notebooks](https://github.com/pybop-team/PyBOP/blob/develop/examples) for hands-on demonstrations:

- [Gravimetric design optimisation (SPMe)](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/design_optimisation/energy_based_electrode_design.ipynb)
- [Non-linear constrained ECM parameter identification](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/ecm_trust-constr.ipynb)
- [Optimiser comparison for parameter identification](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/comparison_examples/multi_optimiser_identification.ipynb)
- [Parameter identification for spatial pouch cell model](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/pouch_cell_identification.ipynb)
- [Estimating ECM parameters from HPPC pulse](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/equivalent_circuit_identification_hppc.ipynb)

### Python Scripts

Find additional script-based examples in the [examples directory](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/):

- [UKF parameter identification (SPM)](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/comparison_examples/unscented_kalman_filter.py)
- [BPX format parameter import/export](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/getting_started/simple_BPX.py)
- [Electrochemical Impendence Spectroscopy (EIS) parameter identification](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/battery_parameterisation/simple_eis.py)
- [Maximum a Posteriori parameter identification (SPM)](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/comparison_examples/maximum_a_posteriori.py)
- [Gradient-based parameter identification (SPM)](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/comparison_examples/adamw.py)


### Supported Methods
The table below lists the currently supported [models](https://github.com/pybop-team/PyBOP/tree/develop/pybop/models), [optimisers](https://github.com/pybop-team/PyBOP/tree/develop/pybop/optimisers), and [cost functions](https://github.com/pybop-team/PyBOP/tree/develop/pybop/costs) in PyBOP.

<p align="center">

| Battery Models                                | Cost Functions                     | Optimization Algorithms                                            |
|-----------------------------------------------|------------------------------------|--------------------------------------------------------------------|
| Single Particle Model (SPM)                   | Sum of Squared Error (SSE)         | Covariance Matrix Adaptation Evolution Strategy (CMA-ES) <tr></tr> |
| Single Particle Model with Electrolyte (SPMe) | Root Mean Squared Error (RMSE)     | Particle Swarm Optimization (PSO) <tr></tr>                        |
| Doyle-Fuller-Newman (DFN)                     | Mean Squared Error (MSE)           | Exponential Natural Evolution Strategy (xNES) <tr></tr>            |
| Many Particle Model (MPM)                     | Mean Absolute Error (MAE)          | Separable Natural Evolution Strategy (sNES)  <tr></tr>             |
| Multi-Species Multi-Reaction (MSMR)           | Minkowski                          | Weight Decayed Adaptive Moment Estimation (AdamW) <tr></tr>        |
| Weppner-Huggins                               | Sum of Power                       | Improved Resilient Backpropagation (iRProp-) <tr></tr>             |
| Equivalent Circuit Models (ECM)               | Gaussian Log Likelihood            | SciPy Minimize & Differential Evolution  <tr></tr>                 |
| Grouped-parameter SPMe (GroupedSPMe)          | Log Posterior                      | Cuckoo Search  <tr></tr>                                           |
|                                               | Gravimetric Energy / Power Density | Simulated Annealing  <tr></tr>                                     |
|                                               | Volumetric Energy / Power Density  | Random Search <tr></tr>                                            |
|                                               |                                    | Gradient Descent <tr></tr>                                         |
|                                               |                                    | Nelder Mead <tr></tr>                                              |

</p>


## Code of Conduct

PyBOP aims to foster a broad consortium of developers and users, building on and learning from the success of the [PyBaMM](https://pybamm.org/) community. Our values are:

-   Inclusivity and fairness (those who wish to contribute may do so, and their input is appropriately recognised)

-   Interoperability (modularity for maximum impact and inclusivity)

-   User-friendliness (putting user requirements first via user-assistance & workflows)

## License

PyBOP is released under the [BSD 3-Clause License](https://github.com/pybop-team/PyBOP/blob/develop/LICENSE).
