import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

"""
This example introduces identification from pulse discharge data to extract:
- Diffusion coefficients governing solid-state lithium transport
- Contact resistance affecting interfacial charge transfer kinetics
The pulse-rest protocol separates kinetic limitations from thermodynamic
equilibrium, enabling extraction of both fast interfacial dynamics and
slower solid-state diffusion processes.
"""

# Define model and parameter values
dfn_model = pybamm.lithium_ion.DFN(options={"contact resistance": "true"})
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"Contact resistance [Ohm]": 0.0045})

# Define the fitting parameters with log transformations for the diffusivities
parameters = [
    pybop.Parameter(
        "Negative particle diffusivity [m2.s-1]",
        prior=pybop.Gaussian(4e-14, 1e-14),
        transformation=pybop.LogTransformation(),
        bounds=[1e-14, 1e-13],
    ),
    pybop.Parameter(
        "Positive particle diffusivity [m2.s-1]",
        prior=pybop.Gaussian(7e-15, 1e-15),
        transformation=pybop.LogTransformation(),
        bounds=[1e-15, 1e-14],
    ),
    pybop.Parameter(
        "Contact resistance [Ohm]",
        initial_value=1e-2,
        bounds=[1e-4, 1e-1],
    ),
]

# Generate a synthetic dataset
sigma = 1e-3
experiment = pybamm.Experiment(
    [
        "Rest for 2 seconds (1 second period)",
        "Discharge at 1C for 1 minute (1 second period)",
        "Rest for 10 minutes (1 second period)",
    ]
)
sim = pybamm.Simulation(
    dfn_model, parameter_values=parameter_values, experiment=experiment
)
sol = sim.solve()

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data + np.random.normal(0, sigma, len(sol.t)),
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Construct the problem builder
spme_model = pybamm.lithium_ion.SPMe(options={"contact resistance": "true"})
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(spme_model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    verbose=True, max_iterations=85, max_unchanged_iterations=25
)
optim = pybop.SimulatedAnnealing(problem, options=options)
optim.optimiser.cooling_rate = 0.825  # Cool quickly due to the low number of iterations
results = optim.run()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_values = results.parameter_values
sim = pybamm.Simulation(
    spme_model, parameter_values=identified_values, experiment=experiment
)
identified_sol = sim.solve()

# Plot identified model vs dataset values
fig, ax = plt.subplots()
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"], label="Target")
ax.plot(identified_sol.t, identified_sol["Voltage [V]"].data, label="Fit")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
ax.legend()
plt.show()

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)
