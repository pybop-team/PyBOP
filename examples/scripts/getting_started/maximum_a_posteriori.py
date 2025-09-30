import os

import numpy as np
import pybamm

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
        "Bulk open-circuit voltage [V]": csv_data[:, 3],
    }
)

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Negative electrode active material volume fraction": 0.43,
        "Positive electrode active material volume fraction": 0.56,
    }
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.8),
        bounds=[0.3, 0.8],
        initial_value=0.653,
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.8),
        bounds=[0.4, 0.7],
        initial_value=0.657,
        transformation=pybop.LogTransformation(),
    ),
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    parameters=parameters,
    protocol=dataset,
    initial_state={"Initial open-circuit voltage [V]": csv_data[0, 2]},
)
likelihood = pybop.GaussianLogLikelihood(dataset)
posterior = pybop.LogPosterior(likelihood)
problem = pybop.Problem(simulator, posterior)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    sigma=0.05,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=50,
)
optim = pybop.XNES(problem, options=options)

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour(steps=10)
