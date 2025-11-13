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
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Negative electrode active material volume fraction": 0.63,
        "Positive electrode active material volume fraction": 0.51,
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.6, 0.05),
            bounds=[0.5, 0.8],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.48, 0.05),
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model,
    parameter_values=parameter_values,
    protocol=dataset,
    initial_state={"Initial open-circuit voltage [V]": csv_data[0, 2]},
)
likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=8e-3)
problem = pybop.Problem(simulator, likelihood)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=50,
)
optim = pybop.XNES(problem, options=options)

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour(bounds=[[0.5, 0.8], [0.4, 0.7]], steps=10)
