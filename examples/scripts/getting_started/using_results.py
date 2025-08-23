import numpy as np
import pybamm
from matplotlib import pyplot as plt

import pybop

"""
In this example, we describe the `pybop.OptimisationResult` object, which provides an interface
to investigate the identification or optimisation performance in additional to providing the
final parameter values in a usable python object.

First, we will set up a simple optimisation workflow.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
corrupt_values = sol["Voltage [V]"](t_eval) + np.random.normal(0, sigma, len(t_eval))
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": corrupt_values,
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    verbose=True,
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.AdamW(problem, options=options)
results = optim.run()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

sim = pybamm.Simulation(model, parameter_values=identified_parameter_values)
identified_sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)

# Plot identified model vs dataset values
fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"])
ax.plot(identified_sol.t, identified_sol["Voltage [V]"].data)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
plt.show()

# We can display more metrics, most of which are
# also included in the `verbose` option within
# the Pints' optimisers
print(f"The starting position: {results.x0}")
print(f"The best cost: {results.best_cost}")
print(f"The identified parameter values: {results.x}")
print(f"The optimisation time:{results.total_runtime} seconds")
