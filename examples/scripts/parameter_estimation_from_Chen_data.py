import pybop
import pandas as pd
import numpy as np
from os import path

# Load dataset
data_path = path.join(pybop.script_path,'..','examples/scripts/Chen_example.csv')
measurements = pd.read_csv(data_path, comment="#").to_numpy()
observations = [
    pybop.Dataset("Time [s]", measurements[:, 0]),
    pybop.Dataset("Current function [A]", measurements[:, 1]),
    pybop.Dataset("Voltage [V]", measurements[:, 2]),
]

# Define model
# parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
model = pybop.models.lithium_ion.SPM()

# Define fitting parameters
params = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.75, 0.05),
        bounds=[0.65, 0.85],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
        bounds=[0.55, 0.75],
    ),
]

# Define optimisation problem
parameterisation = pybop.Optimisation(
    model, observations=observations, fit_parameters=params
)

# Optimise RMSE using NLOpt
results, last_optim, num_evals = parameterisation.rmse(
    signal="Voltage [V]", method="nlopt"
)

# get MAP estimate, starting at a random initial point in parameter space
# optimisation.map(x0=[p.sample() for p in params])

# or sample from posterior
# optimisation.sample(1000, n_chains=4, ....)

# or SOBER
# optimisation.sober()


# Optimisation = pybop.optimisation(model, cost=cost, parameters=parameters, observation=observation)
