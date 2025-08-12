import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pybammeis

import pybop

parameter_values = pybamm.ParameterValues("Chen2020")
options = {"surface form": "differential", "contact resistance": "true"}
model = pybamm.lithium_ion.DFN(options=options)

# Using Pybamm-EIS to generate synthetic data
eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
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
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        bounds=[0.375, 0.775],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Uniform(0.3, 0.9),
        initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
        bounds=[0.375, 0.775],
    ),
]

# Build an SPMe
spme_model = pybamm.lithium_ion.SPMe(options=options)
builder = pybop.builders.PybammEIS()
builder.set_dataset(dataset)
builder.set_simulation(spme_model, parameter_values=parameter_values)
builder.add_cost(pybop.costs.MeanAbsoluteError())
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

options = pybop.PintsOptions(verbose=True)
optim = pybop.XNES(problem, options=options)
results = optim.run()

# Using the fully identified pybamm.ParameterValues object
# we can plot the impedance fit
identified_sim = pybammeis.EISSimulation(
    model, parameter_values=results.parameter_values
)
identified_sol = identified_sim.solve(frequencies)

# Plot
fig, ax = plt.subplots()
ax.plot(sol.real, -sol.imag)  # Synthetic
ax.plot(identified_sol.real, -identified_sol.imag)  # Fit
plt.show()
