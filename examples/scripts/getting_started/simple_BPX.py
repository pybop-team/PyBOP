import json

import numpy as np
import pybamm

import pybop

# Define model
with open("examples/parameters/example_BPX.json") as file:
    parameter_set = pybamm.ParameterValues(json.load(file))
model = pybamm.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle radius [m]",
        prior=pybop.Gaussian(6e-06, 0.1e-6),
        bounds=[1e-6, 9e-6],
        true_value=parameter_set["Negative particle radius [m]"],
    ),
    pybop.Parameter(
        "Positive particle radius [m]",
        prior=pybop.Gaussian(4.5e-07, 0.1e-6),
        bounds=[1e-7, 9e-7],
        true_value=parameter_set["Positive particle radius [m]"],
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 5)
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

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, max_iterations=40, verbose=True)

# Run the optimisation
results = optim.run()
print("True parameters:", parameters.true_value())

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
