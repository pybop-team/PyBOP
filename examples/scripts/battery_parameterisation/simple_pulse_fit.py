import os

import numpy as np
import pybamm

import pybop

# This example introduces pulse fitting
# within PyBOP. 5% SOC pulse data is loaded from a local `csv` file
# and particle diffusivity identification for a SPMe model is performed.
# Additionally, uncertainty metrics are computed.

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "../../data/synthetic/spme_pulse_15.csv")

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
    }
)

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.set_initial_state(f"{csv_data[0, 2]} V")

# Fitting parameters
parameter_values.update(
    {
        "Negative particle diffusivity [m2.s-1]": pybop.Parameter(
            "Negative particle diffusivity [m2.s-1]",
            prior=pybop.Gaussian(4e-14, 1e-14),
            transformation=pybop.LogTransformation(),
            bounds=[1e-14, 1e-13],
        ),
        "Positive particle diffusivity [m2.s-1]": pybop.Parameter(
            "Positive particle diffusivity [m2.s-1]",
            prior=pybop.Gaussian(7e-15, 1e-15),
            transformation=pybop.LogTransformation(),
            bounds=[1e-15, 1e-14],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(model, parameter_values, protocol=dataset)
likelihood = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, likelihood)

# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=100,
    max_unchanged_iterations=30,
)
optim = pybop.CMAES(problem, options=options)

# Slow the step-size shrinking (default is 0.5)
optim.optimiser.eta_min = 0.7

# Run the optimisation
result = optim.run()
print(result)

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour()
