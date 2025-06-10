import matplotlib.pyplot as plt
import numpy as np
import pybamm

import pybop

model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
experiment = pybamm.Experiment(
    [
        "Discharge at 1C for 2 minutes",
        "Charge at 0.1C for 1 minutes",
    ]
)
sim = pybamm.Simulation(model=model, experiment=experiment)
sol = sim.solve()

dataset = pybop.Dataset(
    {
        "Time [s]": sol["Time [s]"].data,
        "Voltage [V]": sol["Voltage [V]"].data,
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Create the builder
builder = pybop.builders.Pybamm()
builder.set_dataset(dataset)
builder.set_simulation(
    model,
    parameter_values=parameter_values,
)
builder.add_parameter(
    {"name": "Negative electrode active material volume fraction", "initial_value": 0.6}
)
builder.add_parameter(
    {"name": "Positive electrode active material volume fraction", "initial_value": 0.6}
)
builder.add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]", 1.0))

# Build the problem
problem = builder.build()

# Solve
problem.set_params(np.array([0.6, 0.6]))
sol = problem.pipeline.solve()

# Plot
fig, ax = plt.subplots()
ax.scatter(sol["Time [s]"].data, sol["Voltage [V]"].data)
plt.show()
