import os

import numpy as np

import pybop

# Get the current directory location and convert to absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_dir, "../../data/synthetic/spm_charge_discharge_75.csv"
)

# Define model and use high-performant solver for sensitivities
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Construct the fitting parameters
# with initial values sampled from a different distribution
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        bounds=[0.3, 0.8],
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        true_value=parameter_set["Positive electrode active material volume fraction"],
        # no bounds
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
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
)
likelihood = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=2e-3 * 1.05)
posterior = pybop.LogPosterior(likelihood)
optim = pybop.IRPropMin(
    posterior, max_iterations=125, max_unchanged_iterations=60, sigma0=0.01
)

results = optim.run()
print(parameters.true_value())

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Contour plot with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)
