import os

import numpy as np

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Construct and update initial parameter values
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.43,
        "Positive electrode active material volume fraction": 0.56,
    }
)

# Define model
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.8),
        bounds=[0.3, 0.8],
        initial_value=0.653,
        true_value=parameter_set["Negative electrode active material volume fraction"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.8),
        bounds=[0.4, 0.7],
        initial_value=0.657,
        true_value=parameter_set["Positive electrode active material volume fraction"],
        transformation=pybop.LogTransformation(),
    ),
)

# Import the synthetic dataset, set model initial state
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
initial_state = {"Initial open-circuit voltage [V]": csv_data[0, 2]}
model.set_initial_state(initial_state=initial_state)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
        "Bulk open-circuit voltage [V]": csv_data[:, 3],
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
)
cost = pybop.LogPosterior(pybop.GaussianLogLikelihood(problem))
optim = pybop.IRPropMin(
    cost,
    sigma0=0.05,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=100,
)

# Run the optimisation
results = optim.run()
print("True parameters:", parameters.true_value())

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape
pybop.plot.contour(cost, steps=15)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.35, 0.7], [0.45, 0.625]])
pybop.plot.contour(optim, bounds=bounds, steps=15)
