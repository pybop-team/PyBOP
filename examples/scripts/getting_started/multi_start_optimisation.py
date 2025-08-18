import numpy as np
import pybamm

import pybop

# This example introduces the multi-start functionality
# in Pybop. Multi-starting the optimisation problem
# allows for an increased likelihood of converging to a
# global optimal location within the search space. This
# is helpful for local optimisation algorithms such as:
# Gradient Descent, AdamW, IRPropMin, ScipyMinimize, etc.

# Define model
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPM()

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.1),
        # bounds=[0.5, 0.8]
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.1),
        # bounds=[0.5, 0.8]
    ),
]

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)
corrupt_values = sol["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate the problem class
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Construct the optimiser with 10 multistart runs
# Each of these runs has a random starting position sampled
# from the parameter priors
options = pybop.PintsOptions(
    sigma=np.asarray([0.6, 0.02]), max_iterations=50, multistart=10, verbose=True
)
optim = pybop.GradientDescent(problem, options=options)

# Run optimisation
results = optim.run()

# We can acquire a pybamm.ParameterValues
# object for the best multi-start
identified_parameter_values = results.parameter_values

# Plot convergence
# pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)

# We can display more metrics, most of which are
# also included in the `verbose` option within
# the Pints' optimisers
print(f"The best starting position: {results.x0}")
print(f"The best cost: {results.best_cost}")
print(f"The best identified parameter values: {results.x}")
print(f"The total optimisation time:{results.time} seconds")
