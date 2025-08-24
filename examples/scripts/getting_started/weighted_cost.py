import numpy as np
import pybamm

import pybop

"""
In this example, we introduce PyBOP's functionality for combining multiple cost into a
single PyBOP problem. This is commonly used for model identification with multiple "signals".
In this case, we identify two parameters based on cell terminal voltage, and average cell
temperature.
"""

# Define model and parameter values
options = {"thermal": "lumped", "contact resistance": "true"}
model = pybamm.lithium_ion.SPMe(options=options)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 2e-2

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

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 300, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
        "X-averaged cell temperature [K]": sol["X-averaged cell temperature [K]"](
            t_eval
        ),
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
options = pybop.PintsOptions(
    sigma=np.asarray([0.01, 1e-4]),
    verbose=True,
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.XNES(problem, options=options)

# Run optimisation
results = optim.run()

# Plot convergence
optim.plot_convergence()

# Plot the parameter traces
optim.plot_parameters()

# Plot the cost landscape with optimisation path
optim.plot_surface()
