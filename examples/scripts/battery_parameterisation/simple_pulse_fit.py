import os

import numpy as np
import pybamm

import pybop

# This example introduces pulse fitting
# within PyBOP. Data is loaded from a local `csv` file
# and particle diffusivity for a DFN model is performed.
# Additionally, uncertainty metrics are computed.

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Define model and use high-performant solver for sensitivities
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=pybamm.IDAKLUSolver())

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
# In this example, we initialise the DFN at the first voltage
# point in `csv_data`, an optimise without rebuilding the
# model on every evaluation.
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    initial_state={"Initial open-circuit voltage [V]": csv_data[0, 2]},
    build_on_evaluation=False,
)

likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=2e-4)

optim = pybop.XNES(
    likelihood,
    verbose=True,
    max_iterations=100,
    max_unchanged_iterations=45,
)

# Run optimisation
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
