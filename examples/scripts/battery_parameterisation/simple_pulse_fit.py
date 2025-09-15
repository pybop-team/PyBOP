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

# Define model and use high-performant solver for sensitivities
parameter_set = pybamm.ParameterValues("Chen2020")
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set,
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle diffusivity [m2.s-1]",
        prior=pybop.Gaussian(4e-14, 1e-14),
        transformation=pybop.LogTransformation(),
        bounds=[1e-14, 1e-13],
    ),
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        prior=pybop.Gaussian(7e-15, 1e-15),
        transformation=pybop.LogTransformation(),
        bounds=[1e-15, 1e-14],
    ),
)

# Import the synthetic dataset
csv_data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": csv_data[:, 0],
        "Current function [A]": csv_data[:, 1],
        "Voltage [V]": csv_data[:, 2],
    }
)

# Generate problem, cost function, and optimisation class
# In this example, we initialise the SPMe at the first voltage
# point in `csv_data`, an optimise without rebuilding the
# model on every evaluation.
initial_state = {"Initial open-circuit voltage [V]": csv_data[0, 2]}
model.set_initial_state(initial_state=initial_state)
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
)

likelihood = pybop.SumSquaredError(problem)
optim = pybop.CMAES(
    likelihood,
    verbose=True,
    sigma0=0.02,
    max_iterations=100,
    max_unchanged_iterations=30,
    # compute_sensitivities=True,
    # n_sensitivity_samples=64,  # Decrease samples for CI (increase for higher accuracy)
)

# Slow the step-size shrinking (default is 0.5)
optim.optimiser.eta_min = 0.7

# Run optimisation
results = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(optim)
