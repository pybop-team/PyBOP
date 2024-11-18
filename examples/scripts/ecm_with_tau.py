import numpy as np
import pybamm

import pybop

"""
An alternative approach to the problem described in
examples/scripts/ecm_tau_constraints.py in which constraints are placed
on tau1 = R1 * C1. Here, tau1 is introduced as a parameter of the model
and C1 is replaced by 1/R1 so that the bounds can be applied directly.
"""

# Define the initial parameter set
parameter_set = pybop.ParameterSet.pybamm("ECM_Example")
parameter_set.update(
    {
        "Initial SoC": 0.75,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.001,
        "R1 [Ohm]": 0.0002,
        "C1 [F]": 10000,
        "Element-1 initial overpotential [V]": 0,
    }
)
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_set.update(
    {
        "R2 [Ohm]": 0.0003,
        "C2 [F]": 5000,
        "Element-2 initial overpotential [V]": 0,
    },
    check_already_exists=False,
)

# Define the model
parameter_set.update(
    {
        "tau1 [s]": parameter_set["R1 [Ohm]"] * parameter_set["C1 [F]"],
        "tau2 [s]": parameter_set["R2 [Ohm]"] * parameter_set["C2 [F]"],
    },
    check_already_exists=False,
)
parameter_set.update(
    {
        "C1 [F]": pybamm.Parameter("tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
        "C2 [F]": pybamm.Parameter("tau2 [s]") / pybamm.Parameter("R2 [Ohm]"),
    }
)
model = pybop.empirical.Thevenin(
    parameter_set=parameter_set,
    options={"number of rc elements": 2},
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.0002, 0.0001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.0001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
    pybop.Parameter(
        "tau1 [s]",
        prior=pybop.Gaussian(1.0, 0.025),
        bounds=[0, 3.0],
    ),
)

sigma = 0.001
t_eval = np.arange(0, 600, 3)
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
optim = pybop.XNES(
    cost,
    sigma0=[1e-4, 1e-4, 0.02],  # Set parameter specific step size
    allow_infeasible_solutions=False,
    max_unchanged_iterations=30,
    max_iterations=125,
)

results = optim.run()
print(
    "True parameters:",
    [
        parameter_set["R0 [Ohm]"],
        parameter_set["R1 [Ohm]"],
        parameter_set["tau1 [s]"],
    ],
    [parameter_set.evaluate(pybamm.Parameter("C1 [F]"))],
)
print("Estimated parameters:", results.x.tolist(), [results.x[2] / results.x[1]])

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)
