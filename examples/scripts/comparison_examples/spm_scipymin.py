import pandas as pd

import pybop

# Form dataset
Measurements = pd.read_csv("examples/data/Chen_example.csv", comment="#").to_numpy()
dataset = pybop.Dataset(
    {
        "Time [s]": Measurements[:, 0],
        "Current function [A]": Measurements[:, 1],
        "Voltage [V]": Measurements[:, 2],
    }
)

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.models.lithium_ion.SPM(
    parameter_set=parameter_set, options={"thermal": "lumped"}
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.7],
    ),
)

# Define the cost to optimise
signal = ["Voltage [V]"]
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.SciPyMinimize(cost)

# Run the optimisation problem
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
