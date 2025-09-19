import numpy as np
import pybamm

import pybop

"""
An alternative approach to the problem described in
examples/scripts/ecm_tau_constraints.py in which constraints are placed
on tau1 = R1 * C1. Here, tau1 is introduced as a parameter of the model
and C1 is replaced by 1/R1 so that the bounds can be applied directly.
"""

# Define model
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Define the initial parameter set
parameter_values = pybamm.ParameterValues("ECM_Example")
parameter_values.update(
    {
        "Initial SoC": 0.75,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": model.default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.001,
        "R1 [Ohm]": 0.0002,
        "C1 [F]": 10000,
        "Element-1 initial overpotential [V]": 0,
    }
)
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_values.update(
    {
        "R2 [Ohm]": 0.0003,
        "C2 [F]": 5000,
        "Element-2 initial overpotential [V]": 0,
    },
    check_already_exists=False,
)
parameter_values.update(
    {
        "tau1 [s]": parameter_values["R1 [Ohm]"] * parameter_values["C1 [F]"],
        "tau2 [s]": parameter_values["R2 [Ohm]"] * parameter_values["C2 [F]"],
    },
    check_already_exists=False,
)
parameter_values.update(
    {
        "C1 [F]": pybamm.Parameter("tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
        "C2 [F]": pybamm.Parameter("tau2 [s]") / pybamm.Parameter("R2 [Ohm]"),
    }
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

# Generate a synthetic dataset
sigma = 0.001
t_eval = np.arange(0, 600, 3)
sol = pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval=t_eval)
corrupt_values = sol["Voltage [V]"](t_eval) + np.random.normal(0, sigma, len(t_eval))
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": sol["Current [A]"](t_eval),
        "Voltage [V]": corrupt_values,
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values, input_parameter_names=parameters.names, protocol=dataset
)
problem = pybop.FittingProblem(simulator, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)

# Set up the optimiser
options = pybop.PintsOptions(
    sigma=[1e-4, 1e-4, 0.02],  # Set parameter specific step size
    max_unchanged_iterations=30,
    max_iterations=125,
)
optim = pybop.XNES(cost, options=options)

# Run the optimisation
result = optim.run()
print(result)
print(
    "True parameters:",
    [
        parameter_values["R0 [Ohm]"],
        parameter_values["R1 [Ohm]"],
        parameter_values["tau1 [s]"],
        parameter_values.evaluate(pybamm.Parameter("C1 [F]")),
    ],
)
print("Estimated parameters:", result.x.tolist() + [result.x[2] / result.x[1]])

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
pybop.plot.convergence(optim)
pybop.plot.parameters(optim)
