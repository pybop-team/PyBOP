import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle radius [m]",
        prior=pybop.Gaussian(6e-06, 0.1e-6),
        bounds=[1e-6, 9e-6],
        true_value=parameter_set["Negative particle radius [m]"],
        transformation=pybop.LogTransformation(),
    ),
    pybop.Parameter(
        "Positive particle radius [m]",
        prior=pybop.Gaussian(4.5e-06, 0.1e-6),
        bounds=[1e-6, 9e-6],
        true_value=parameter_set["Positive particle radius [m]"],
        transformation=pybop.LogTransformation(),
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 5)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
        "Bulk open-circuit voltage [V]": values["Bulk open-circuit voltage [V]"].data,
    }
)

signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, sigma0=0.25, max_unchanged_iterations=10, max_iterations=40)

# Run the optimisation
results = optim.run()
print(results)
print("True parameters:", parameters.true_value())

# Plot the time series
pybop.plot.dataset(dataset)

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape
pybop.plot.surface(optim)
