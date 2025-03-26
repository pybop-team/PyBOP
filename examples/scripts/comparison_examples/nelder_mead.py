import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
    ),
)

# Generate data
sigma = 0.003
experiment = pybop.Experiment(
    [
        "Discharge at 0.5C for 3 minutes (3 second period)",
        "Charge at 0.5C for 3 minutes (3 second period)",
    ]
    * 2
)
values = model.predict(initial_state={"Initial SoC": 0.5}, experiment=experiment)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": noisy(values["Voltage [V]"].data, sigma),
        "Bulk open-circuit voltage [V]": noisy(
            values["Bulk open-circuit voltage [V]"].data, sigma
        ),
    }
)

# Generate problem, cost function, and optimisation class
signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    signal=signal,
    initial_state={"Initial open-circuit voltage [V]": dataset["Voltage [V]"][0]},
)
cost = pybop.RootMeanSquaredError(problem)
optim = pybop.NelderMead(
    cost,
    verbose=True,
    allow_infeasible_solutions=True,
    sigma0=0.05,
    max_iterations=100,
    max_unchanged_iterations=20,
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
