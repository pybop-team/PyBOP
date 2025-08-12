import numpy as np
import pybamm

import pybop

# In this example, we introduce Pybop's functionality
# for combining multiple cost into a single Pybop
# problem. This is commonly used for model identification
# with multiple "signals". In this case, we identify
# two parameters based on cell terminal voltage, and
# average cell temperature.

# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 2e-2
model = pybamm.lithium_ion.SPMe(
    options={"thermal": "lumped", "contact resistance": "true"}
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Contact resistance [Ohm]",
        initial_value=1e-2,
        bounds=[1e-4, 5e-2],
    ),
]

# Generate the synthetic dataset
sigma = 5e-3
t_eval = np.linspace(0, 300, 240)
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
        "X-averaged cell temperature [K]": sol["X-averaged cell temperature [K]"].data,
    }
)

# Construct the problem builder, with weighting on each cost
# Each cost represents a different observed signal in this example
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
    .add_cost(
        pybop.costs.pybamm.SumOfPower(
            "X-averaged cell temperature [K]",
            "X-averaged cell temperature [K]",
            p=np.sqrt(2),
        ),
        weight=1 / 273.15,
    )
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
# We use the Improved Backpropagation Plus implementation
# This is a gradient-based optimiser, with a step-size
# which is decoupled from the gradient magnitude
options = pybop.PintsOptions(
    sigma=np.asarray([0.01, 1e-4]),
    verbose=True,
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.XNES(problem, options=options)
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
