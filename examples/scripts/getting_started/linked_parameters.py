import numpy as np
import pybamm

import pybop

# In this example, we introduce the functionality
# to link optimisation parameters for the underlying
# Pybamm model. Linking parameters can ensure correlated
# parameters are correctly updated, which ensures
# physical definitions are maintained. For this example
# we link the electrode porosity, active material volume fraction
# and binder percentage.

# Define model and parameter values
# Link the porosity in both electrodes
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"Positive electrode binder": 0.02}, check_already_exists=False)
parameter_values.update({"Negative electrode binder": 0.02}, check_already_exists=False)
parameter_values["Positive electrode porosity"] = (
    1.0
    - parameter_values["Positive electrode active material volume fraction"]
    - parameter_values["Positive electrode porosity"]
)
parameter_values["Negative electrode porosity"] = (
    1.0
    - parameter_values["Negative electrode active material volume fraction"]
    - parameter_values["Negative electrode porosity"]
)
model = pybamm.lithium_ion.SPM()

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
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
# We use the Improved Backpropagation Plus implementation
# This is a gradient-based optimiser, with a step-size
# which is decoupled from the gradient magnitude
options = pybop.PintsOptions(
    sigma=0.1, verbose=True, max_iterations=60, max_unchanged_iterations=15
)
optim = pybop.CMAES(problem, options=options)
results = optim.run()

# Obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes
identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(problem, steps=20)
