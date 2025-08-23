import numpy as np
import pybamm

import pybop

"""
This example presents the process of creating a multi-fitting problem.
The multi-fitting problem allows for multiple problems to be optimised
at the same time, common use cases include:
- Fitting multiple datasets for a single model (varying SOC identification)
- Fitting different models for the same dataset (comparing reduced-order implementations)

Note: the optimisation parameters have to be the same for each problem.

In this example, we will identify parameters on the same model for two different datasets.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Create list of initial SOC and experiment
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
datasets = []
for i in [0, 1]:
    sim = pybamm.Simulation(
        model, experiment=experiment[i], parameter_values=parameter_values
    )
    sol = sim.solve(initial_soc=init_soc[i])
    dataset_i = pybop.Dataset(
        {
            "Time [s]": sol.t,
            "Current function [A]": sol["Current [A]"].data,
            "Voltage [V]": sol["Voltage [V]"].data
            + np.random.normal(0, sigma, len(sol.t)),
        }
    )
    datasets.append(dataset_i)

# Construct the problem class
builder = (
    pybop.builders.Pybamm()
    .set_dataset(datasets[0])
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.MeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem1 = builder.build()

# Update builder and build the second problem
builder.set_dataset(datasets[1])
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
