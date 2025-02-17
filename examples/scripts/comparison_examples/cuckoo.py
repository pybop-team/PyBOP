import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.7,
        "Positive electrode active material volume fraction": 0.67,
    }
)
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.4, 0.75],
        initial_value=0.41,
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.75],
        initial_value=0.41,
    ),
)
experiment = pybop.Experiment(
    [
        "Rest for 5 seconds (1 second period)",
        "Discharge at 0.5C for 3 minutes (4 second period)",
        "Charge at 0.5C for 3 minutes (4 second period)",
    ]
)
values = model.predict(experiment=experiment, initial_state={"Initial SoC": 0.65})

sigma = 0.002
corrupt_values = values["Voltage [V]"].data + np.random.normal(
    0, sigma, len(values["Voltage [V]"].data)
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(
    model,
    parameters,
    dataset,
    initial_state={"Initial open-circuit voltage [V]": dataset["Voltage [V]"][0]},
)
# cost = pybop.GaussianLogLikelihood(problem, sigma0=sigma * 4)
cost = pybop.SumSquaredError(problem)
optim = pybop.XNES(
    cost,
    max_iterations=100,
)

results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
