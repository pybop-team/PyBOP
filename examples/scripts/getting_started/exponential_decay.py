import numpy as np
import pybamm

import pybop

# Define model and use high-performant solver for sensitivities
parameter_set = pybamm.ParameterValues({"k": 1, "y0": 0.5})
model = pybop.ExponentialDecayModel(parameter_set=parameter_set, n_states=2)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "k",
        prior=pybop.Gaussian(0.5, 0.05),
    ),
    pybop.Parameter(
        "y0",
        prior=pybop.Gaussian(0.2, 0.05),
    ),
)

# Generate data
sigma = 0.003
t_eval = np.linspace(0, 10, 100)
values = model.predict(t_eval=t_eval)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": 0 * t_eval,
        "y_0": noisy(values["y_0"].data, sigma),
        "y_1": noisy(values["y_1"].data, sigma),
    }
)

signal = ["y_0", "y_1"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.Minkowski(problem, p=2)
optim = pybop.AdamW(
    cost,
    verbose=True,
    allow_infeasible_solutions=True,
    sigma0=0.02,
    max_iterations=100,
    max_unchanged_iterations=20,
)

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
