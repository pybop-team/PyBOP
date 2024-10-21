import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        true_value=parameter_set["Positive electrode active material volume fraction"],
    ),
)

# Generate a dataset and a fitting problem
sigma = 0.001
experiment = pybop.Experiment([("Discharge at 0.5C for 2 minutes (4 second period)")])
values = model.predict(initial_state={"Initial SoC": 0.8}, experiment=experiment)
dataset_1 = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data
        + np.random.normal(0, sigma, len(values["Voltage [V]"].data)),
    }
)
problem_1 = pybop.FittingProblem(model, parameters, dataset_1)

# Generate a second dataset and problem
model = model.new_copy()
experiment = pybop.Experiment([("Discharge at 1C for 1 minutes (4 second period)")])
values = model.predict(initial_state={"Initial SoC": 0.8}, experiment=experiment)
dataset_2 = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data
        + np.random.normal(0, sigma, len(values["Voltage [V]"].data)),
    }
)
problem_2 = pybop.FittingProblem(model, parameters, dataset_2)

# Combine the problems into one
problem = pybop.MultiFittingProblem(problem_1, problem_2)

# Generate the cost function and optimisation class
cost = pybop.SumSquaredError(problem)
optim = pybop.IRPropMin(
    cost,
    verbose=True,
    max_iterations=100,
)

# Run optimisation
results = optim.run()
print("True parameters:", parameters.true_value())
print("Estimated parameters:", results.x)

# Plot the timeseries output
pybop.plot.quick(problem_1, problem_inputs=results.x, title="Optimised Comparison")
pybop.plot.quick(problem_2, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
