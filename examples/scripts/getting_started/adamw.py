import os

import numpy as np
import pybamm

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Define model and use high-performant solver for sensitivities
parameter_set = pybamm.ParameterValues("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        initial_value=0.45,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        initial_value=0.45,
        bounds=[0.4, 0.9],
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

signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    signal=signal,
)
cost = pybop.SumOfPower(problem, p=2.5)

optim = pybop.AdamW(
    cost,
    verbose=True,
    verbose_print_rate=20,
    allow_infeasible_solutions=True,
    sigma0=0.02,
    max_iterations=100,
    max_unchanged_iterations=45,
    compute_sensitivities=True,
    n_sensitivity_samples=128,
)
# Reduce the momentum influence
# for the reduced number of optimiser iterations
optim.optimiser.b1 = 0.9
optim.optimiser.b2 = 0.9

# Run optimisation
results = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
