import numpy as np
import pybamm

import pybop

"""
In this example, we introduce Simulated Annealing, a probabilistic optimisation method
inspired by the annealing process in metallurgy. It works by iteratively proposing new
solutions and accepting them based on both their fitness and a temperature parameter that
decreases over time. This allows the algorithm to initially explore broadly and gradually
focus on local optimisation as the temperature decreases.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.4, 0.9],
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 2e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    sigma=5e-2, verbose=True, max_iterations=120, max_unchanged_iterations=60
)
optim = pybop.SimulatedAnnealing(problem, options=options)

# Update the cooling rate to bias towards better solutions (lower exploration)
optim.optimiser.cooling_rate = 0.85

# Run optimisation
results = optim.run()

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
