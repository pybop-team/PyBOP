import numpy as np
import pybamm

import pybop

"""
Example of parameter estimation using an equivalent circuit model.
"""

# Define the model
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Import the ECM parameter values from a JSON file
# parameter_values = pybamm.ParameterValues.create_from_bpx(
#     "examples/parameters/initial_ecm_parameters.json"
# )

# Alternatively, define the initial parameter values via a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_values = pybamm.ParameterValues(
    {
        "chemistry": "ecm",
        "Initial SoC": 0.5,
        "Initial temperature [K]": 25 + 273.15,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Ambient temperature [K]": 25 + 273.15,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Cell thermal mass [J/K]": 1000,
        "Cell-jig heat transfer coefficient [W/K]": 10,
        "Jig thermal mass [J/K]": 500,
        "Jig-air heat transfer coefficient [W/K]": 10,
        "Open-circuit voltage [V]": model.default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "Entropic change [V/K]": 0.0004,
        "R0 [Ohm]": 0.001,
        "Element-1 initial overpotential [V]": 0,
        "R1 [Ohm]": 0.002,
        "R2 [Ohm]": 0.003,
        "Element-2 initial overpotential [V]": 0,
        "C1 [F]": 1000,
        "C2 [F]": 500,
    }
)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.002, 0.001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.001, 0.001),
        bounds=[1e-5, 1e-2],
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.arange(0, 500, 3)
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
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Construct optimiser with additional options
options = pybop.PintsOptions(max_iterations=250)
optim = pybop.PSO(problem, options=options)

# Run optimisation
results = optim.run()
print(results)

# Plot the cost landscape with optimisation path
results.plot_surface()

# Compare to known values
print("True parameters:", [parameter_values[p.name] for p in parameters])
print(f"Idenitified Parameters: {results.x}")

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, dataset=dataset)
