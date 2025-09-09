import matplotlib.pyplot as plt
import pybamm

import pybop

# This example introduces the Grouped SPMe, a model which groups the correlated parameters
# of the Single Particle Model with Electrolyte (SPMe) into a minimum set of parameters for
# identification purposes.

experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (5 seconds period)",
        "Rest for 10 minutes (5 seconds period)",
    ],
)

# Use the Chen2020 parameters with a constant electrolyte conductivity
param = pybamm.ParameterValues("Chen2020")
ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
T = param["Initial temperature [K]"]
param["Electrolyte conductivity [S.m-1]"] = param["Electrolyte conductivity [S.m-1]"](
    ce0, T
)
param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](
    ce0, T
)

# Enable options
model_options = {"surface form": "differential", "contact resistance": "true"}

# Solve the PyBaMM SPMe
model = pybamm.lithium_ion.SPMe(options=model_options)
simulation = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
sol = simulation.solve()

# To set the grouped parameters, we can use the mapping method provided
# by the GroupedSPMe class
grouped_param = pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(
    parameter_values=param
)
print(grouped_param)

# Solve the Grouped SPMe
model = pybop.lithium_ion.GroupedSPMe(options=model_options)
simulation = pybamm.Simulation(
    model, parameter_values=grouped_param, experiment=experiment
)
sol2 = simulation.solve()

# Plot Models
fig, ax = plt.subplots()
ax.plot(sol.t, sol["Voltage [V]"].data, label="PyBaMM SPMe")
ax.plot(sol2.t, sol2["Voltage [V]"].data, "--", label="Grouped SPMe")
ax.legend()
fig.show()
