import matplotlib.pyplot as plt
import pybamm

import pybop

# This example introduces the Grouped SPMe model
# which groups correlated parameters into a minimum
# set for the Single Particle Model with Electrolyte (SPMe).
# This allows for a minimal set of parameters
# for identification purposes.

experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V (5 seconds period)",
        "Rest for 10 minutes (5 seconds period)",
    ],
)

# Set ParameterValues, the GroupedSPMe uses the Chen2020 as the basis
parameter_values = pybop.lithium_ion.GroupedSPMe().default_parameter_values

# To see the reduced parameter values, we can use to mapping method provided
# by the GroupedSPMe class.
parameter_set = pybamm.ParameterValues("Chen2020")
grouped_parameters = pybop.lithium_ion.GroupedSPMe().create_grouped_parameters(
    parameter_set
)
print(grouped_parameters)

# Solve the Grouped SPMe
model_options = {"surface form": "differential", "contact resistance": "true"}
model = pybop.lithium_ion.GroupedSPMe(options=model_options)
simulation = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
)
sol = simulation.solve()

# Solve the PyBaMM SPMe
model_options = {"surface form": "differential", "contact resistance": "true"}
model = pybamm.lithium_ion.SPMe(options=model_options)
parameter_values = pybamm.ParameterValues("Chen2020")
simulation = pybamm.Simulation(
    model, parameter_values=parameter_values, experiment=experiment
)
sol2 = simulation.solve()

# Plot Models
fig, ax = plt.subplots()
ax.plot(sol.t, sol["Voltage [V]"].data)
ax.plot(sol2.t, sol2["Voltage [V]"].data)
fig.show()
