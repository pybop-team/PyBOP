import numpy as np
import pybamm
from matplotlib import pyplot as plt

import pybop

# In this example, we describe the `pybop.OptimiserResults`
# object, which provides an interface to investigate
# the identification or optimisation performance
# in additional to providing the final parameter values
# in a usable python object.

# First, we will set up a simple optimisation workflow
# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPM()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
]

# Generate the synthetic dataset
sigma = 5e-3
t_eval = np.linspace(0, 500, 240)
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)
corrupt_values = sol["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": corrupt_values,
        "Current function [A]": sol["Current [A]"].data,
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
# We use the Nelder-Mead simplex based
# optimiser for this example. Additionally,
# the step-size value (sigma) is increased to 0.1
# to search across the landscape further per iteration
options = pybop.PintsOptions(
    sigma=0.1,
    verbose=True,
    max_iterations=60,
    max_unchanged_iterations=15,
    multistart=3,
)
optim = pybop.AdamW(problem, options=options)
results = optim.run()


# Now we have a results object. The first thing we can
# do is obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes. Since ran the
# optimiser with multi-starts, this object is a list of the identified
# parameter values for each run.
identified_parameter_values = results.parameter_values

# We can also get the `best` parameter values. This is selected based
# on comparing the error-measure convergence (i.e. the lowest final cost
# for canonical costs (SSE, RMSE, etc.) and the largest likelihood for
# likelihood functions).
sim = pybamm.Simulation(model, parameter_values=results.parameter_values_best)
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
print(f"The list of starting positions: {results.x0}")
print(f"The list of final costs: {results.final_cost}")
print(f"The list of identified parameter values: {results.x}")
print(f"The list of optimisation times:{results.time} seconds")

# Similarly, the `best` attribute is available for most metrics
print(f"The starting positions for the best optimisation: {results.x0_best}")
print(f"The final cost for the best optimisation: {results.final_cost_best}")
print(f"The best identified parameter values: {results.x_best}")
print(f"The time for the best optimisation:{results.time_best} seconds")
