import numpy as np
import pybamm

import pybop

# This example presents the process of creating a multi-fitting problem
# The multi-fitting problem allows for multiple problems to be optimised
# at the same time, common use cases include:
# - Fitting multiple datasets for a single model (varying SOC identification)
# - Fitting different models for the same dataset (comparing reduced-order implementations)

# Note: the optimisation parameters have to be the same for each problem

# In this example, we will identify parameters on the same model
# for two different datasets.

# Parameter values and model definition
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPM()

# Create initial SOC, experiment objects
init_soc = [0.8, 0.6]
experiment = [
    pybamm.Experiment([("Discharge at 0.5C for 2 minutes (6 second period)")]),
    pybamm.Experiment([("Discharge at 1C for 1 minutes (6 second period)")]),
]

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.58,
        bounds=[0.5, 0.8],
    ),
]

# Generate the first dataset
sigma = 0.002
sim = pybamm.Simulation(
    model, experiment=experiment[0], parameter_values=parameter_values
)
sol1 = sim.solve(initial_soc=init_soc[0])
dataset_1 = pybop.Dataset(
    {
        "Time [s]": sol1["Time [s]"].data,
        "Current function [A]": sol1["Current [A]"].data,
        "Voltage [V]": sol1["Voltage [V]"].data
        + np.random.normal(0, sigma, len(sol1["Voltage [V]"].data)),
    }
)

# Generate the second dataset
sim = pybamm.Simulation(
    model, experiment=experiment[1], parameter_values=parameter_values
)
sol2 = sim.solve(initial_soc=init_soc[1])
dataset_2 = pybop.Dataset(
    {
        "Time [s]": sol2["Time [s]"].data,
        "Current function [A]": sol2["Current [A]"].data,
        "Voltage [V]": sol2["Voltage [V]"].data
        + np.random.normal(0, sigma, len(sol2["Voltage [V]"].data)),
    }
)

# Construct the problem class
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset_1)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.MeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem1 = builder.build()

# Update builder and build the second problem
builder.set_dataset(dataset_2)
problem2 = builder.build()

# Create a multifitting builder, with weighting applied to the first model
multi_builder = (
    pybop.builders.MultiFitting()
    .add_problem(problem1, weight=0.75)
    .add_problem(problem2)
)
multi_problem = multi_builder.build()

# Construct the optimiser with additional options
options = pybop.PintsOptions(
    verbose=True, sigma=0.05, max_unchanged_iterations=20, max_iterations=100
)
optim = pybop.CMAES(multi_problem, options=options)

# Run optimisation
results = optim.run()

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
