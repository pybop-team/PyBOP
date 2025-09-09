import numpy as np
import pybamm

import pybop

"""
In this example, we introduce the Adaptive Moment Estimation with Weight Decay (AdamW)
optimisation algorithm. This optimiser uses gradient information for trajectory and
step-size determination.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
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
    .add_cost(pybop.costs.pybamm.MeanSquaredError("Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    sigma=0.01, verbose=True, max_iterations=200, max_unchanged_iterations=100
)
optim = pybop.AdamW(problem, options=options)

# Reduce the momentum influence for the reduced number of optimiser iterations
optim.optimiser.b1 = 0.925
optim.optimiser.b2 = 0.925

# Run optimisation
results = optim.run()

# Plot convergence
results.plot_convergence()

# Plot the parameter traces
results.plot_parameters()

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.6, 0.9], [0.5, 0.8]])
results.plot_surface(bounds=bounds)
