import numpy as np
import pybamm

import pybop

# In this example, we introduce a simple Doyle-Fuller-Newman
# model identification process. This is specifically implemented
# as the DFN is a more challenging identification process compared
# to the reduced-order single particle models.

# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.DFN()

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
        "Bulk open-circuit voltage [V]": sol["Bulk open-circuit voltage [V]"].data,
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
    .add_cost(
        pybop.costs.pybamm.RootMeanSquaredError(
            "Bulk open-circuit voltage [V]", "Bulk open-circuit voltage [V]"
        )
    )
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
    sigma=0.1, verbose=True, max_iterations=60, max_unchanged_iterations=15
)
optim = pybop.NelderMead(problem, options=options)
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
