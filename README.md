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

PyBOP provides tools for the parameterisation and optimisation of battery models, using both Bayesian and frequentist approaches, with [example workflows](https://github.com/pybop-team/PyBOP/tree/main/examples/) to assist the user. PyBOP can be used to parameterise various battery models, including the electrochemical and equivalent circuit models available in [PyBaMM](https://pybamm.org/).

📌 PyBOP v25.10 presents a [major restructure](https://github.com/pybop-team/PyBOP/blob/main/CHANGELOG.md) of PyBOP's base classes. We move from setting up
a model, problem, cost, then optimiser to defining a simulator, cost, problem, and then optimiser. A `pybop.pybamm.Simulator` is designed to simulate a
`pybamm.BaseModel`. Optimisation parameters can be passed through a `pybamm.ParameterValues` class.
To understand how to update your use of PyBOP, please take a look at the example notebooks and scripts.

<p align="center">
    <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/PyBOP-high-level.svg" alt="pybop_arch.svg" width="700" />
</p>

## 💻 Installation

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


## 💡 Use Cases

PyBOP has two intended uses:

1. Parameter inference from battery test data.

2. Design optimisation under battery manufacturing/use constraints.

These include a wide variety of optimisation problems that require careful consideration due to the choice of battery model, data availability and/or the choice of design parameters.

### Publications

Please take inspiration from the following journal articles which show how PyBOP is being used for research:

- "Physics-based battery model parametrisation from impedance data" by [Hallemans et al. (2025)](https://doi.org/10.1149/1945-7111/add41b) with [open-source code and data](https://github.com/Battery-Intelligence-Lab/Hallemans-2025-JES)

### Jupyter Notebooks

Explore our [example notebooks](https://github.com/pybop-team/PyBOP/tree/develop/examples) for hands-on demonstrations:

- [Getting started with gradient-based optimisation](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/getting_started/optimising_with_adamw.ipynb)
- [Estimating ECM parameters from a HPPC pulse](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/ecm_hppc_identification.ipynb)
- [Identifying ECM parameters with nonlinear constraints](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/ecm_scipy_constraints.ipynb)
- [Parameter identification for a spatial pouch cell model](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/battery_parameterisation/pouch_cell_identification.ipynb)
- [Energy-based electrode design optimisation](https://nbviewer.org/github/pybop-team/PyBOP/blob/develop/examples/notebooks/design_optimisation/energy_based_electrode_design.ipynb)

### Python Scripts

Find additional script-based examples in the [examples directory](https://github.com/pybop-team/PyBOP/tree/develop/examples/scripts):

- [Getting started with SciPy minimize](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/getting_started/optimising_with_scipy_minimize.py)
- [Estimating diffusivity from GITT data](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/battery_parameterisation/gitt_fitting.py)
- [Maximum a Posteriori parameter identification](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/getting_started/maximum_a_posteriori.py)
- [Using electrochemical impedance spectroscopy (EIS) data](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/battery_parameterisation/simple_eis.py)
- [Getting started with MCMC samplers](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/getting_started/monte_carlo_sampling.py)


### Grouped Models
In addition to the models available in PyBaMM, PyBOP currently hosts some grouped-parameter versions of common battery models which are
purpose-built for parameter esimation and can be found under [models](https://github.com/pybop-team/PyBOP/tree/develop/pybop/models).


## 👉 Code of Conduct

PyBOP aims to foster a broad consortium of developers and users, building on and learning from the success of the [PyBaMM](https://pybamm.org/) community. Our values are:

-   Inclusivity and fairness (those who wish to contribute may do so, and their input is appropriately recognised)

-   Interoperability (modularity for maximum impact and inclusivity)

-   User-friendliness (putting user requirements first via user-assistance & workflows)

## 📃 License

PyBOP is released under the [BSD 3-Clause License](https://github.com/pybop-team/PyBOP/blob/develop/LICENSE).

## 🌟 Contributing

We would like to thank all contributors to PyBOP. Contributions are welcome! See [CONTRIBUTING.md](https://github.com/pybop-team/PyBOP/blob/develop/CONTRIBUTING.md) for ways to get started.

The original PyBOP developers were supported by research funding from the [Faraday Institution](https://www.faraday.ac.uk/),
[UK Research and Innovation](https://www.ukri.org/),
and [Horizon Europe IntelLiGent Consortium](https://heuintelligent.eu).
