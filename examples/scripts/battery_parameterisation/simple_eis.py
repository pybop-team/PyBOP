import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pybammeis

import pybop

"""
Example demonstrating parameter estimation in the frequency domain using PyBaMM-EIS.
"""

# Define model and parameter values
options = {"surface form": "differential", "contact resistance": "true"}
dfn_model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Chen2020")

# Using Pybamm-EIS to generate synthetic data
eis_sim = pybammeis.EISSimulation(dfn_model, parameter_values=parameter_values)
frequencies = np.logspace(-4, 4, 60)
sol = eis_sim.solve(frequencies)

dataset = pybop.Dataset(
    {
        "Frequency [Hz]": frequencies,
        "Current function [A]": np.ones(frequencies.size) * 0.0,
        "Impedance": sol,
    }
)

parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        bounds=[0.375, 0.775],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        bounds=[0.375, 0.775],
    ),
]

# Construct the problem builder using an SPMe
spme_model = pybamm.lithium_ion.SPMe(options=options)
builder = pybop.builders.PybammEIS()
builder.set_dataset(dataset)
builder.set_simulation(spme_model, parameter_values=parameter_values)
builder.add_cost(pybop.costs.MeanAbsoluteError())
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(max_iterations=250)
optim = pybop.XNES(problem, options=options)

# Run optimisation
results = optim.run()
print(results)

# Compare to known values
print("True parameters:", [parameter_values[p.name] for p in parameters])
print(f"Idenitified Parameters: {results.x}")

# Using the identified pybamm.ParameterValues object, we can plot the impedance fit
identified_parameter_values = results.parameter_values
identified_sim = pybammeis.EISSimulation(dfn_model, parameter_values=parameter_values)
identified_sol = identified_sim.solve(frequencies)

# Plot comparison
fig, ax = plt.subplots()
ax = pybammeis.nyquist_plot(sol, ax=ax, label="Target", alpha=0.7)
ax = pybammeis.nyquist_plot(identified_sol, ax=ax, label="Fit", marker="x", alpha=0.7)
ax.legend()
plt.show()
