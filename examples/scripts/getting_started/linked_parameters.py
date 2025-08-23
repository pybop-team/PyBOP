import numpy as np
import pybamm
from pybamm import Parameter

import pybop

"""
In this example, we introduce the functionality to link optimisation parameters for the
underlying PyBaMM model. Linking parameters can ensure correlated parameters are correctly
updated, which ensures physical definitions are maintained. For this example, we link the
electrode porosity, active material volume fraction and binder percentage.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Link the porosity in both electrodes
parameter_values.update(
    {
        "Positive electrode porosity": (
            1.0
            - Parameter("Positive electrode active material volume fraction")
            - Parameter("Positive electrode binder fraction")
        ),
        "Negative electrode porosity": (
            1.0
            - Parameter("Negative electrode active material volume fraction")
            - Parameter("Negative electrode binder fraction")
        ),
    }
)
parameter_values.update(
    {
        "Positive electrode binder fraction": 0.02,
        "Negative electrode binder fraction": 0.02,
    },
    check_already_exists=False,
)

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
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options. We use the Improved Backpropagation Plus implementation
# This is a gradient-based optimiser, with a step-size which is decoupled from the
# gradient magnitude
options = pybop.PintsOptions(
    sigma=0.1, verbose=True, max_iterations=60, max_unchanged_iterations=15
)
optim = pybop.CMAES(problem, options=options)
results = optim.run()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(problem, steps=15)
