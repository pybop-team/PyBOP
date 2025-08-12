import numpy as np
import pybamm

import pybop

# In this example, we introduce the
# Adaptive Moment Estimation with Weight Decay (AdamW)
# optimisation algorithm. This optimiser uses gradient
# information for trajectory and step-size determination.

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

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.MeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    sigma=0.01, verbose=True, max_iterations=200, max_unchanged_iterations=100
)
optim = pybop.AdamW(problem, options=options)
# Reduce the momentum influence
# for the reduced number of optimiser iterations
optim.optimiser.b1 = 0.925
optim.optimiser.b2 = 0.925

results = optim.run()

# Obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes
identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
