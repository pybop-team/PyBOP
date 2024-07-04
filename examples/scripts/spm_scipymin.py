import pandas as pd

import pybop

# Form dataset
Measurements = pd.read_csv("examples/scripts/Chen_example.csv", comment="#").to_numpy()
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
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal, init_soc=0.98)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.SciPyMinimize(cost)

# Run the optimisation problem
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot2d(optim, steps=15)
