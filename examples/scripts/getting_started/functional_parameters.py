import numpy as np
import pybamm

import pybop

"""
This example demonstrates how to use a pybamm.FunctionalParameter to
optimise functional parameters using PyBOP.

Method: Define a new scalar parameter for use in a functional parameter
that already exists in the model, for example an exchange current density.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM(options={"contact resistance": "true"})
parameter_values = pybamm.ParameterValues("Chen2020")


# Define a new function using pybamm parameters
def positive_electrode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    # New parameters
    j0_ref = pybamm.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]"
    )
    alpha = pybamm.Parameter("Positive electrode charge transfer coefficient")

    # Existing parameters
    c_e_init = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        j0_ref
        * (c_s_surf / c_s_max) ** alpha
        * ((c_e / c_e_init) * (1 - c_s_surf / c_s_max)) ** (1 - alpha)
    )


# Give default values to the new scalar parameters and pass the new function
parameter_values.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": 1,
        "Positive electrode charge transfer coefficient": 0.5,
    },
    check_already_exists=False,
)
parameter_values["Positive electrode exchange-current density [A.m-2]"] = (
    positive_electrode_exchange_current_density
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]",
        initial_value=1.2,
        bounds=[0.75, 1.25],
    ),
    pybop.Parameter(
        "Positive electrode charge transfer coefficient",
        initial_value=0.3,
        bounds=[0.25, 0.75],
    ),
]

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
options = pybop.PintsOptions(sigma=0.01, max_iterations=100, verbose=True)
optim = pybop.NelderMead(problem, options=options)

# Run optimisation
results = optim.run()

# Plot convergence
results.plot_convergence()

# Plot the parameter traces
results.plot_parameters()

# Plot the cost landscape with optimisation path
results.plot_surface()

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, dataset=dataset)
