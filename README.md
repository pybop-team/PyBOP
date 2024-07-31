<div align="center">

  <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/logo/PyBOP_logo_flat.png" alt="logo.svg" width="700" />

  # Python Battery Optimisation and Parameterisation


  [![Scheduled](https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml/badge.svg)](https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml)
  [![Contributors](https://img.shields.io/github/contributors/pybop-team/PyBOP)](https://github.com/pybop-team/PyBOP/graphs/contributors)
  [![Last Commit](https://img.shields.io/github/last-commit/pybop-team/PyBOP/develop?color=purple)](https://github.com/pybop-team/PyBOP/commits/develop)
  [![Python Versions from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpybop-team%2FPyBOP%2Fdevelop%2Fpyproject.toml&label=Python)](https://pypi.org/project/pybop/)
  [![Forks](https://img.shields.io/github/forks/pybop-team/PyBOP?style=flat)](https://github.com/pybop-team/PyBOP/network/members)
  [![Stars](https://img.shields.io/github/stars/pybop-team/PyBOP?style=flat&color=gold)](https://github.com/pybop-team/PyBOP/stargazers)
  [![Codecov](https://codecov.io/gh/pybop-team/PyBOP/branch/develop/graph/badge.svg)](https://codecov.io/gh/pybop-team/PyBOP)
  [![Open Issues](https://img.shields.io/github/issues/pybop-team/PyBOP)](https://github.com/pybop-team/PyBOP/issues/)
  [![License](https://img.shields.io/github/license/pybop-team/PyBOP?color=blue)](https://github.com/pybop-team/PyBOP/blob/develop/LICENSE)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybop-team/PyBOP/blob/develop/)
  [![Static Badge](https://img.shields.io/badge/https%3A%2F%2Fpybop-team.github.io%2Fpybop-bench%2F?label=Benchmarks)](https://pybop-team.github.io/pybop-bench/)
  [![Releases](https://img.shields.io/github/v/release/pybop-team/PyBOP?color=gold)](https://github.com/pybop-team/PyBOP/releases)

</div>

## PyBOP
PyBOP provides a complete set of tools for parameterisation and optimisation of battery models, using both Bayesian and frequentist approaches, with [example workflows](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks) to assist the user. PyBOP can be used to parameterise various battery models, including electrochemical and equivalent circuit models available in [PyBaMM](https://pybamm.org/). PyBOP prioritises clear and informative diagnostics for the user, while also allowing for advanced probabilistic methods.

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

1. Parameter estimation from battery test data.

2. Design optimisation under battery manufacturing/use constraints.

These include a wide variety of optimisation problems that require careful consideration due to the choice of battery model, data availability and/or the choice of design parameters.

### Notebooks
PyBOP comes with a number of [example notebooks](https://github.com/pybop-team/PyBOP/blob/develop/examples), which can be found in the examples folder. A few noteworthy ones are listed below.

- [Gravimetric design optimisation of a single particle model](https://github.com/pybop-team/PyBOP/blob/develop/examples/notebooks/spm_electrode_design.ipynb)
- [Experimental GITT fitting of an ECM for an LG M50](https://github.com/pybop-team/PyBOP/blob/develop/examples/notebooks/LG_M50_ECM/1-single-pulse-circuit-model.ipynb)
- [Compare PyBOP optimisers for parameter identification](https://github.com/pybop-team/PyBOP/blob/develop/examples/notebooks/multi_optimiser_identification.ipynb)
- [Parameter identification for a spatial pouch cell model](https://github.com/pybop-team/PyBOP/blob/develop/examples/notebooks/pouch_cell_identification.ipynb)

### Scripts
Additional script-based examples can be found in the [examples directory](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/). Some notable scripts are listed below.

- [Unscented Kalman filter parameter identification of a SPM](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/spm_UKF.py)
- [Import and export parameters using Faraday's BPX format](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/BPX_spm.py)
- [Maximum a posteriori parameter identification of a SPM](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/BPX_spm.py)
- [Gradient based parameter identification of a SPM](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/spm_AdamW.py)

### Supported Methods
The table below lists the currently supported [models](https://github.com/pybop-team/PyBOP/tree/develop/pybop/models), [optimisers](https://github.com/pybop-team/PyBOP/tree/develop/pybop/optimisers), and [cost functions](https://github.com/pybop-team/PyBOP/tree/develop/pybop/costs) in PyBOP.

<p align="center">

| Battery Models | Optimization Algorithms | Cost Functions |
|-----------------------------------------------|-------------------------------------------------------------|------------------------------------------|
| Single Particle Model (SPM) | Covariance Matrix Adaptation Evolution Strategy (CMA-ES) | Sum of Squared Errors (SSE) <tr></tr> |
| Single Particle Model with Electrolyte (SPMe) | Particle Swarm Optimization (PSO) | Root Mean Squared Error (RMSE) <tr></tr> |
| Doyle-Fuller-Newman (DFN) | Exponential Natural Evolution Strategy (xNES)  | Gaussian Log Likelihood <tr></tr> |
| Many Particle Model (MPM) | Separable Natural Evolution Strategy (sNES) | Gaussian Log Likelihood w/ known variance <tr></tr> |
| Multi-Species Multi-Reactants (MSMR) | Adaptive Moment Estimation with Weight Decay (AdamW) | Maximum a Posteriori (MAP) <tr></tr> |
| Equivalent Circuit Models (ECM) | Improved Resilient Backpropagation (iRProp-) | Unscented Kalman Filter (UKF) <tr></tr> |
| | SciPy Minimize & Differential Evolution | Gravimetric Energy Density <tr></tr> |
| | Gradient Descent| Volumetric Energy Density<tr></tr> |
| | Nelder-Mead | <tr></tr> |

</p>


## Code of Conduct

PyBOP aims to foster a broad consortium of developers and users, building on and learning from the success of the [PyBaMM](https://pybamm.org/) community. Our values are:

-   Inclusivity and fairness (those who wish to contribute may do so, and their input is appropriately recognised)

-   Interoperability (modularity for maximum impact and inclusivity)

-   User-friendliness (putting user requirements first via user-assistance & workflows)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://bradyplanden.github.io"><img src="https://avatars.githubusercontent.com/u/55357039?v=4?s=100" width="100px;" alt="Brady Planden"/><br /><sub><b>Brady Planden</b></sub></a><br /><a href="#infra-BradyPlanden" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=BradyPlanden" title="Tests">⚠️</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=BradyPlanden" title="Code">💻</a> <a href="#example-BradyPlanden" title="Examples">💡</a> <a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3ABradyPlanden" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NicolaCourtier"><img src="https://avatars.githubusercontent.com/u/45851982?v=4?s=100" width="100px;" alt="NicolaCourtier"/><br /><sub><b>NicolaCourtier</b></sub></a><br /><a href="https://github.com/pybop-team/PyBOP/commits?author=NicolaCourtier" title="Code">💻</a> <a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3ANicolaCourtier" title="Reviewed Pull Requests">👀</a> <a href="#example-NicolaCourtier" title="Examples">💡</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=NicolaCourtier" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://howey.eng.ox.ac.uk"><img src="https://avatars.githubusercontent.com/u/2247552?v=4?s=100" width="100px;" alt="David Howey"/><br /><sub><b>David Howey</b></sub></a><br /><a href="#ideas-davidhowey" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-davidhowey" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.rse.ox.ac.uk"><img src="https://avatars.githubusercontent.com/u/1148404?v=4?s=100" width="100px;" alt="Martin Robinson"/><br /><sub><b>Martin Robinson</b></sub></a><br /><a href="#ideas-martinjrobins" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-martinjrobins" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3Amartinjrobins" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=martinjrobins" title="Code">💻</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=martinjrobins" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.brosaplanella.xyz"><img src="https://avatars.githubusercontent.com/u/28443643?v=4?s=100" width="100px;" alt="Ferran Brosa Planella"/><br /><sub><b>Ferran Brosa Planella</b></sub></a><br /><a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3Abrosaplanella" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=brosaplanella" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/agriyakhetarpal"><img src="https://avatars.githubusercontent.com/u/74401230?v=4?s=100" width="100px;" alt="Agriya Khetarpal"/><br /><sub><b>Agriya Khetarpal</b></sub></a><br /><a href="https://github.com/pybop-team/PyBOP/commits?author=agriyakhetarpal" title="Code">💻</a> <a href="#infra-agriyakhetarpal" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3Aagriyakhetarpal" title="Reviewed Pull Requests">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://faraday.ac.uk"><img src="assets/faraday-logo.jpg?s=100" width="100px;" alt="Faraday Institution"/><br /><sub><b>Faraday Institution</b></sub></a><br /><a href="#financial-FaradayInstitution" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.ukri.org/"><img src="assets/UKRI.png?s=100" width="100px;" alt="UK Research and Innovation"/><br /><sub><b>UK Research and Innovation</b></sub></a><br /><a href="#financial-UKRI" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://heuintelligent.eu/"><img src="assets/logo-farger.svg" width="100px;" alt="EU IntelLiGent Project"/><br /><sub><b>IntelLiGent Consortium</b></sub></a><br /><a href="#financial-UKRI" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/muhammedsogut/"><img src="https://avatars.githubusercontent.com/u/34511375?v=4?s=100" width="100px;" alt="Muhammed Nedim Sogut"/><br /><sub><b>Muhammed Nedim Sogut</b></sub></a><br /><a href="https://github.com/pybop-team/PyBOP/commits?author=muhammedsogut" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specifications. Contributions of any kind are welcome! See `CONTRIBUTING.md` for ways to get started.
