import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

options = {"surface form": "differential", "contact resistance": "true"}
dfn_model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.001

dataset = pybop.Dataset(
    {
        "Frequency [Hz]": np.logspace(-4, 5, 100),
        "Current function [A]": np.ones(100) * 0.0,
        "Impedance": np.ones(100) * 0.0,
    }
)

# Create the builder
builder = pybop.builders.PybammEIS()
builder.set_dataset(dataset)
builder.set_simulation(dfn_model, parameter_values=parameter_values, initial_state=0.5)
builder.add_parameter(
    pybop.Parameter("Negative electrode thickness [m]", initial_value=70e-6)
)
builder.add_parameter(
    pybop.Parameter("Positive particle radius [m]", initial_value=4e-6)
)
builder.add_cost(pybop.SumSquaredError())

# Build the DFN problem
problem = builder.build()

inputs = np.asarray([80e-6, 4.5e-6])
problem.set_params(inputs)
sol1 = problem.pipeline.solve()
val1 = problem.run()

# Build an SPMe
spme_model = pybamm.lithium_ion.SPMe(options=options)
builder.set_simulation(spme_model, parameter_values=parameter_values, initial_state=0.5)
problem = builder.build()

# Solve
problem.set_params(inputs)
sol2 = problem.pipeline.solve()

# Plot
fig, ax = plt.subplots()
ax.plot(sol1.real, -sol1.imag)  # DFN
ax.plot(sol2.real, -sol2.imag)  # SPMe
plt.show()
