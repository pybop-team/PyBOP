import pybop
import pandas as pd

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
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.75, 0.05),
        bounds=[0.6, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.65, 0.05),
        bounds=[0.5, 0.8],
    ),
]

# Define the cost to optimise
signal = "Voltage [V]"
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal, init_soc=0.98)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
optim = pybop.Optimisation(cost=cost, optimiser=pybop.NLoptOptimize)

# Run the optimisation problem
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(x, cost, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape
pybop.plot_cost2d(cost, steps=15)

# Plot the cost landscape with optimisation path
pybop.plot_cost2d(cost, optim=optim, steps=15)
