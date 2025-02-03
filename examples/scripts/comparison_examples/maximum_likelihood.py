import numpy as np
import pybamm

import pybop

# Define model and set initial parameter values
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.63,
        "Positive electrode active material volume fraction": 0.51,
    }
)
solver = pybamm.IDAKLUSolver()
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set, solver=solver)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
    ),
)

# Generate data
sigma = 0.001
experiment = pybop.Experiment(
    [
        (
            "Discharge at 0.5C for 3 minutes (3 second period)",
            "Charge at 0.5C for 3 minutes (3 second period)",
        ),
    ]
)
values = model.predict(initial_state={"Initial SoC": 0.15}, experiment=experiment)


def noise(sigma):
    return np.random.normal(0, sigma, len(values["Voltage [V]"].data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data + noise(sigma),
        "Bulk open-circuit voltage [V]": values["Bulk open-circuit voltage [V]"].data
        + noise(sigma),
    }
)


signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
likelihood = pybop.GaussianLogLikelihood(problem, sigma0=sigma * 4)
optim = pybop.CMAES(
    likelihood,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=100,
)

# Run the optimisation
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.55, 0.77], [0.48, 0.68]])
pybop.plot.contour(optim, bounds=bounds, steps=15)
