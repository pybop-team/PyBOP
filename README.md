<div align="center">
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

  <img src="assets/Temp_Logo.png" alt="logo" width="400" height="auto" />
  <h1>Python Battery Optimisation and Parameterisation</h1>


<p>
  <a href="https://github.com/pybop-team/PyBOP/actions/workflows/test_on_push.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/pybop-team/PyBOP/test_on_push.yaml?label=Build%20Status" alt="build" />
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
  <a href="https://github.com/pybop-team/PyBOP/issues/">
    <img src="https://img.shields.io/github/issues/pybop-team/PyBOP" alt="open issues" />
  </a>
  <a href="https://github.com/pybop-team/PyBOP/blob/develop/LICENSE">
    <img src="https://img.shields.io/github/license/pybop-team/PyBOP" alt="license" />
  </a>
</p>

</div>

<!-- Software Specification -->
## PyBOP
PyBOP provides a comprehensive suite of tools for parameterisation and optimisation of battery models. It aims to implement Bayesian and frequentist techniques with example workflows to guide the user. PyBOP can be applied to parameterise a wide range of battery models, including the electrochemical and equivalent circuit models available in PyBAMM. A major emphasis in PyBOP is understandable and actionable diagnostics for the user, while still providing extensibility for advanced probabilistic methods. By building on the state-of-the-art battery models and leveraging Python's accessibility, PyBOP enables agile and robust parameterisation and optimisation.

The figure below gives PyBOP's current conceptual structure. The living software specification of PyBOP can be found [here](https://github.com/pybop-team/software-spec). This package is under active development, expect API evolution with releases.


<p align="center">
    <img src="assets/PyBOP_Arch.svg" alt="Data flows from battery cycling machines to Galv Harvesters, then to the     Galv server and REST API. Metadata can be updated and data read using the web client, and data can be downloaded by the Python client." width="600" />
</p>

<!-- Getting Started -->
## Getting Started

<!-- Installation -->
### Installation

Create a virtual environment, i.e with [pyenv](https://github.com/pyenv/pyenv#installation) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installation):

```bash
pyenv virtualenv pybop-env
pyenv activate pybop-env
```

Install PyBOP:

```bash
 pip install git+https://github.com/pybop-team/PyBOP
```

<!-- Installation -->
### Usage
The example below shows a simple fitting routine that starts by generating synthetic data from a single particle model with modified parameter values. An RMSE cost function using the terminal voltage as the optimised signal is completed to determine the unknown parameter values.

```python
import pybop
import pybamm
import pandas as pd
import numpy as np

def getdata(x0):
        model = pybamm.lithium_ion.SPM()
        params = model.default_parameter_values

        params.update(
            {
                "Negative electrode active material volume fraction": x0[0],
                "Positive electrode active material volume fraction": x0[1],
            }
        )
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 2C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
        return sim.solve()


# Form observations
x0 = np.array([0.55, 0.63])
solution = getdata(x0)

observations = [
    pybop.Observed("Time [s]", solution["Time [s]"].data),
    pybop.Observed("Current function [A]", solution["Current [A]"].data),
    pybop.Observed("Voltage [V]", solution["Terminal voltage [V]"].data),
]

# Define model
model = pybop.models.lithium_ion.SPM()
model.parameter_set = model.pybamm_model.default_parameter_values

# Fitting parameters
params = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.5, 0.05),
        bounds=[0.35, 0.75],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
        bounds=[0.45, 0.85],
    ),
]

parameterisation = pybop.Parameterisation(
    model, observations=observations, fit_parameters=params
)

# get RMSE estimate using NLOpt
results, last_optim, num_evals = parameterisation.rmse(
    signal="Voltage [V]", method="nlopt" # results = [0.54452026, 0.63064801]
)
```

<!-- Code of Conduct -->
## Code of Conduct

PyBOP aims to foster a broad consortium of developers and users, building on and
learning from the success of the PyBaMM community. Our values are:

-   Open-source (code and ideas should be shared)

-   Inclusivity and fairness (those who want to contribute may do so, and their input is appropriately recognised)

-   Inter-operability (aiming for modularity to enable maximum impact and inclusivity)

-   User-friendliness (putting user requirements first, thinking about user-assistance & workflows)


<!-- Contributing -->
## Contributing
Thanks to all of our contributing members! [[emoji key](https://allcontributors.org/docs/en/emoji-key)]

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://bradyplanden.github.io"><img src="https://avatars.githubusercontent.com/u/55357039?v=4?s=100" width="100px;" alt="Brady Planden"/><br /><sub><b>Brady Planden</b></sub></a><br /><a href="#infra-BradyPlanden" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=BradyPlanden" title="Tests">⚠️</a> <a href="https://github.com/pybop-team/PyBOP/commits?author=BradyPlanden" title="Code">💻</a> <a href="#example-BradyPlanden" title="Examples">💡</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Contributions are always welcome! See `contributing.md` for ways to get started.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!