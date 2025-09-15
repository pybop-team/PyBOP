import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybamm.ParameterValues("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Create initial SOC, experiment objects
init_soc = [{"Initial SoC": 0.8}, {"Initial SoC": 0.6}]
experiment = [
    pybamm.Experiment([("Discharge at 0.5C for 2 minutes (4 second period)")]),
    pybamm.Experiment([("Discharge at 1C for 1 minutes (4 second period)")]),
]

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
sigma = 0.002
values = model.predict(initial_state=init_soc[0], experiment=experiment[0])
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
values = model.predict(initial_state=init_soc[1], experiment=experiment[1])
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
optim = pybop.CuckooSearch(
    cost,
    verbose=True,
    sigma0=0.05,
    max_unchanged_iterations=20,
    max_iterations=100,
)

# Run optimisation
results = optim.run()
print("True parameters:", parameters.true_value())

# Plot the timeseries output
pybop.plot.problem(problem_1, problem_inputs=results.x, title="Optimised Comparison")
pybop.plot.problem(problem_2, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
