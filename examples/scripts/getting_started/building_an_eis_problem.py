import numpy as np
import pybamm
import pybammeis

import pybop

"""
Example demonstrating how to build an EIS optimisation problem.
"""

options = {"surface form": "differential", "contact resistance": "true"}
model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Chen2020")

# Using PyBaMM-EIS to generate synthetic data
eis_sim = pybammeis.EISSimulation(model, parameter_values=parameter_values)
frequencies = np.logspace(-4, 4, 30)
sol = eis_sim.solve(frequencies)

dataset = pybop.Dataset(
    {
        "Frequency [Hz]": frequencies,
        "Current function [A]": np.ones(frequencies.size) * 0.0,
        "Impedance": sol,
    }
)

# Create the builder
builder = pybop.builders.PybammEIS()
builder.set_dataset(dataset)
builder.set_simulation(model, parameter_values=parameter_values)
builder.add_parameter(
    pybop.Parameter("Negative electrode thickness [m]", initial_value=70e-6)
)
builder.add_parameter(
    pybop.Parameter("Positive particle radius [m]", initial_value=4e-6)
)
builder.add_cost(pybop.SumSquaredError())

# Build the DFN problem
problem = builder.build()

# Test the cost for a given parameter proposal
cost = problem.run(np.asarray([80e-6, 4.5e-6]))
print("The cost value is:", cost)
