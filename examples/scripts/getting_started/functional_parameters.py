import numpy as np
import pybamm

import pybop

# This example demonstrates how to use a pybamm.FunctionalParameter to
# optimise functional parameters using PyBOP.

# Method: Define a new scalar parameter for use in a functional parameter
# that already exists in the model, for example an exchange current density.


# Load parameter set
parameter_set = pybop.ParameterSet().pybamm("Chen2020")


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
        * ((c_e / c_e_init) * (c_s_surf / c_s_max) * (1 - c_s_surf / c_s_max)) ** alpha
    )


# Give default values to the new scalar parameters and pass the new function
parameter_set.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": 1,
        "Positive electrode charge transfer coefficient": 0.5,
    },
    check_already_exists=False,
)
parameter_set["Positive electrode exchange-current density [A.m-2]"] = (
    positive_electrode_exchange_current_density
)

# Model definition
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set, options={"contact resistance": "true"}
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]",
        prior=pybop.Gaussian(1, 0.1),
    ),
    pybop.Parameter(
        "Positive electrode charge transfer coefficient",
        prior=pybop.Gaussian(0.5, 0.1),
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)
optim = pybop.SciPyMinimize(cost, sigma0=0.1, max_iterations=125)

# Run optimisation
results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
