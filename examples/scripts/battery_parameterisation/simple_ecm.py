import json

import numpy as np
import pybamm

import pybop

# Define the model
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Import the ECM parameter set from JSON
with open("examples/parameters/initial_ecm_parameters.json") as file:
    parameter_values = pybamm.ParameterValues(json.load(file))
parameter_values.update(
    {
        "Open-circuit voltage [V]": model.default_parameter_values[
            "Open-circuit voltage [V]"
        ]
    },
    check_already_exists=False,
)

# Alternatively, define the initial parameter set with a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
# parameter_values = pybamm.ParameterValues(
#     params_dict={
#         "chemistry": "ecm",
#         "Initial SoC": 0.5,
#         "Initial temperature [K]": 25 + 273.15,
#         "Cell capacity [A.h]": 5,
#         "Nominal cell capacity [A.h]": 5,
#         "Ambient temperature [K]": 25 + 273.15,
#         "Current function [A]": 5,
#         "Upper voltage cut-off [V]": 4.2,
#         "Lower voltage cut-off [V]": 3.0,
#         "Cell thermal mass [J/K]": 1000,
#         "Cell-jig heat transfer coefficient [W/K]": 10,
#         "Jig thermal mass [J/K]": 500,
#         "Jig-air heat transfer coefficient [W/K]": 10,
#         "Open-circuit voltage [V]": model.default_parameter_values[
#             "Open-circuit voltage [V]"
#         ],
#         "R0 [Ohm]": 0.001,
#         "Element-1 initial overpotential [V]": 0,
#         "Element-2 initial overpotential [V]": 0,
#         "R1 [Ohm]": 0.0002,
#         "R2 [Ohm]": 0.0003,
#         "C1 [F]": 10000,
#         "C2 [F]": 5000,
#         "Entropic change [V/K]": 0.0004,
#     }
# )

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
        bounds=[1e-5, 1e-3],
    ),
)

# Generate a synthetic dataset
sigma = 0.001
t_eval = np.arange(0, 900, 3)
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
    model,
    parameter_values=parameter_values,
    parameters=parameters,
    protocol=dataset,
)
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(max_iterations=100)
optim = pybop.CMAES(problem, options=options)

# Run the optimisation
result = optim.run()
print(result)
print("True values:", [parameter_values[p] for p in parameters.keys()])

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
