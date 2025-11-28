import numpy as np
import pybamm

import pybop

"""
This example demonstrates how to use a pybamm.FunctionParameter to optimise
functional parameters using PyBOP.

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

# Generate a synthetic dataset
sigma = 0.001
t_eval = np.arange(0, 900, 3)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=t_eval
)
corrupt_values = solution["Voltage [V]"](t_eval) + np.random.normal(
    0, sigma, len(t_eval)
)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": solution["Current [A]"](t_eval),
        "Voltage [V]": corrupt_values,
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(1, 0.1),
        ),
        "Positive electrode charge transfer coefficient": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(0.5, 0.1),
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.RootMeanSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.SciPyMinimizeOptions(maxiter=125, verbose=True)
optim = pybop.SciPyMinimize(problem, options=options)

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0, 2], [0, 1]])
