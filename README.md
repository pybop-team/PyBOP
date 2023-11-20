<div align="center">

  <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/Temp_Logo.png" alt="logo" width="400" height="auto" />
  <h1>Python Battery Optimisation and Parameterisation</h1>

<p>
  <a href="https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml">
    <img src="https://github.com/pybop-team/PyBOP/actions/workflows/scheduled_tests.yaml/badge.svg" alt="Scheduled" />
  </a>
  <a href="https://github.com/pybop-team/PyBOP/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/pybop-team/PyBOP" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/pybop-team/PyBOP/develop" alt="last update" />
  </a>
  <a href="https://github.com/pybop-team/PyBOPe/network/members">
    <img src="https://img.shields.io/github/forks/pybop-team/PyBOP" alt="forks" />
  </a>
  <a href="https://github.com/pybop-team/PyBOP/stargazers">
    <img src="https://img.shields.io/github/stars/pybop-team/PyBOP" alt="stars" />
  </a>
  <a href="https://codecov.io/gh/pybop-team/PyBOP">
    <img src="https://codecov.io/gh/pybop-team/PyBOP/branch/develop/graph/badge.svg" alt="codecov" />
  </a>
  <a href="https://github.com/pybop-team/PyBOP/issues/">
    <img src="https://img.shields.io/github/issues/pybop-team/PyBOP" alt="open issues" />
  </a>
  <a href="https://github.com/pybop-team/PyBOP/blob/develop/LICENSE">
    <img src="https://img.shields.io/github/license/pybop-team/PyBOP" alt="license" />
  </a>
  <a href="https://colab.research.google.com/github/pybop-team/PyBOP/blob/develop/">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</p>

</div>

<!-- Software Specification -->
## PyBOP
PyBOP offers a full range of tools for the parameterisation and optimisation of battery models, utilising both Bayesian and frequentist approaches with example workflows to assist the user. PyBOP can be used to parameterise various battery models, which include electrochemical and equivalent circuit models that are present in [PyBaMM](https://pybamm.org/). PyBOP prioritises clear and informative diagnostics for users, while also allowing for advanced probabilistic methods.

The diagram below presents PyBOP's conceptual framework. The PyBOP software specification is available at [this link](https://github.com/pybop-team/software-spec). This product is currently undergoing development, and users can expect the API to evolve with future releases.

<p align="center">
    <img src="https://raw.githubusercontent.com/pybop-team/PyBOP/develop/assets/PyBOP_Architecture.png" alt="Data flows from battery cycling machines to Galv Harvesters, then to the     Galv server and REST API. Metadata can be updated and data read using the web client, and data can be downloaded by the Python client." width="600" />
</p>

<!-- Getting Started -->
## Getting Started

<!-- Installation -->
### Prerequisites
To use and/or contribute to PyBOP, first install Python (3.8-3.11). On a Debian-based distribution, this looks like:

```bash
sudo apt update
sudo apt install python3 python3-virtualenv
```

For further information, please refer to the similar [installation instructions for PyBaMM](https://docs.pybamm.org/en/latest/source/user_guide/installation/GNU-linux.html).

### Installation

Create a virtual environment called `pybop-env` within your current directory:

```bash
virtualenv pybop-env
```

Activate the environment:

```bash
source pybop-env/bin/activate
```

Later, you can deactivate the environment:

```bash
deactivate
```

Within your virtual environment, install the `develop` branch of PyBOP:

```bash
pip install git+https://github.com/pybop-team/PyBOP.git@develop
```

To alternatively install PyBOP from a local directory, use the following template, substituting in the relevant path:

```bash
pip install -e "PATH_TO_PYBOP"
```

To check whether PyBOP has been installed correctly, run one of the examples in the following section or the full set of unit tests:

```bash
pytest --unit -v
```

### Using PyBOP
PyBOP has two general types of intended use cases:
1. parameter estimation from battery test data
2. design optimisation subject to battery manufacturing/usage constraints

These general cases encompass a wide variety of optimisation problems that require careful consideration based on the choice of battery model, the available data and/or the choice of design parameters.

PyBOP comes with a number of [example notebooks and scripts](https://github.com/pybop-team/PyBOP/blob/develop/examples) which can be found in the examples folder.

The [spm_descent.py](https://github.com/pybop-team/PyBOP/blob/develop/examples/scripts/spm_descent.py) script illustrates a straightforward example that starts by generating artificial data from a single particle model (SPM). The unknown parameter values are identified by implementing a sum-of-square error cost function using the terminal voltage as the observed signal and a gradient descent optimiser. To run this example:

```bash
python examples/scripts/spm_descent.py
```

In addition, [spm_nlopt.ipynb](https://github.com/pybop-team/PyBOP/blob/develop/examples/notebooks/spm_nlopt.ipynb) provides a second example in notebook form. This example estimates the SPM parameters based on an RMSE cost function and a BOBYQA optimiser.

<!-- Code of Conduct -->
## Code of Conduct

PyBOP aims to foster a broad consortium of developers and users, building on and learning from the success of the [PyBaMM](https://pybamm.org/) community. Our values are:

-   Inclusivity and fairness (those who wish to contribute may do so, and their input is appropriately recognised)

-   Interoperability (modularity for maximum impact and inclusivity)

-   User-friendliness (putting user requirements first via user-assistance & workflows)


<!-- Contributing -->
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
      <td align="center" valign="top" width="14.28%"><a href="http://www.rse.ox.ac.uk"><img src="https://avatars.githubusercontent.com/u/1148404?v=4?s=100" width="100px;" alt="Martin Robinson"/><br /><sub><b>Martin Robinson</b></sub></a><br /><a href="#ideas-martinjrobins" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-martinjrobins" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3Amartinjrobins" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.brosaplanella.xyz"><img src="https://avatars.githubusercontent.com/u/28443643?v=4?s=100" width="100px;" alt="Ferran Brosa Planella"/><br /><sub><b>Ferran Brosa Planella</b></sub></a><br /><a href="https://github.com/pybop-team/PyBOP/pulls?q=is%3Apr+reviewed-by%3Abrosaplanella" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specifications. Contributions of any kind are welcome! See `contributing.md` for ways to get started.
