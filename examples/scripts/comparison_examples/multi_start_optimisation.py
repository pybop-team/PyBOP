import numpy as np
import pybamm

import pybop

"""
This example introduces the multi-start functionality in PyBOP. Multi-starting the
optimisation problem allows for an increased likelihood of converging to a global
optimal location within the search space. This is helpful for local optimisation
algorithms such as: Gradient Descent, AdamW, IRPropMin, ScipyMinimize, etc.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.arange(0, 900, 3)
sol = sim.solve(t_eval=t_eval)

sigma = 0.001
corrupt_values = sol["Voltage [V]"](t_eval) + np.random.normal(0, sigma, len(t_eval))
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": sol["Current [A]"](t_eval),
        "Voltage [V]": corrupt_values,
    }
)

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

# Construct the optimiser with 10 multistart runs. Each of these runs has a random
# starting position sampled from the parameter priors
options = pybop.PintsOptions(
    sigma=np.asarray([0.6, 0.02]), max_iterations=50, multistart=10
)
optim = pybop.GradientDescent(problem, options=options)

# Run optimisation
results = optim.run()
print(results)

# We can display more metrics, most of which are also included in the `verbose` option
# within the PINTS optimisers
print(f"The best starting position: {results.x0}")
print(f"The best cost: {results.best_cost}")
print(f"The best identified parameter values: {results.x}")
print(f"The total optimisation time:{results.time} seconds")

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)
