import pybop
import pandas as pd
import matplotlib.pyplot as plt

# Form dataset
Measurements = pd.read_csv("examples/scripts/Chen_example.csv", comment="#").to_numpy()
dataset = [
    pybop.Dataset("Time [s]", Measurements[:, 0]),
    pybop.Dataset("Current function [A]", Measurements[:, 1]),
    pybop.Dataset("Terminal voltage [V]", Measurements[:, 2]),
]

# Define model
parameter_set = pybop.ParameterSet("pybamm", "Chen2020")
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
signal = "Terminal voltage [V]"
problem = pybop.FittingProblem(parameters, dataset, model, signal=signal, init_soc=0.98)
cost = pybop.RootMeanSquaredError(problem)

# Build the optimisation problem
parameterisation = pybop.Optimisation(cost=cost, optimiser=pybop.NLoptOptimize)

# Run the optimisation problem
x, final_cost = parameterisation.run()

# Show the generated data
simulated_values = problem.evaluate(x)

plt.figure()
plt.xlabel("Time")
plt.ylabel("Values")
plt.plot(dataset[0].data, dataset[2].data, label="Measured")
plt.plot(dataset[0].data, simulated_values, label="Simulated")
plt.legend(bbox_to_anchor=(0.6, 1), loc="upper left", fontsize=12)
plt.show()
