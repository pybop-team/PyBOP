import time

import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode conductivity [S.m-1]",
        prior=pybop.Gaussian(150, 10),
        bounds=[100, 300],
    ),
    pybop.Parameter(
        "Positive electrode conductivity [S.m-1]",
        prior=pybop.Gaussian(0.5, 0.1),
        bounds=[0.05, 1],
    ),
]

sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)
# Create list of dict of inputs to simulate
j = [
    {
        "Negative electrode conductivity [S.m-1]": x * 30,
        "Positive electrode conductivity [S.m-1]": x / 10,
    }
    for x in range(1, 9)
]

# Generate problem, call model.simulate() and time it
problem = pybop.FittingProblem(model, parameters, dataset)

time_start = time.time()
problem.model.simulate(t_eval=t_eval, inputs=j)
time_end = time.time()

print("Time taken to evaluate the problem: ", time_end - time_start)
