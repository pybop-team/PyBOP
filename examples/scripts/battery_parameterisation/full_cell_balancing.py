import numpy as np
import pybamm

import pybop

"""
In this example, we present a method for full-cell stoichiometry balancing. This is
completed by identifying the initial and maximum concentrations in each electrode
using low-rate discharge observations.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Upper voltage cut-off [V]"] = 4.4
parameter_values["Lower voltage cut-off [V]"] = 2.3

# Set initial state and unpack true values
parameter_values.set_initial_state(1.0)
cs_n_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
cs_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
cs_n_init = parameter_values["Initial concentration in negative electrode [mol.m-3]"]
cs_p_init = parameter_values["Initial concentration in positive electrode [mol.m-3]"]

# Define fitting parameters for OCP balancing
parameters = [
    pybop.Parameter(
        "Maximum concentration in negative electrode [mol.m-3]",
        initial_value=cs_n_max * 0.8,
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
    ),
    pybop.Parameter(
        "Maximum concentration in positive electrode [mol.m-3]",
        initial_value=cs_p_max * 0.8,
        bounds=[cs_p_max * 0.75, cs_p_max * 1.25],
    ),
    pybop.Parameter(
        "Initial concentration in negative electrode [mol.m-3]",
        initial_value=cs_n_max * 0.8,
        bounds=[cs_n_max * 0.75, cs_n_max * 1.25],
    ),
    pybop.Parameter(
        "Initial concentration in positive electrode [mol.m-3]",
        initial_value=cs_p_max * 0.2,
        bounds=[0, cs_p_max * 0.5],
    ),
]

# Generate a synthetic dataset
sigma = 1e-3
experiment = pybamm.Experiment(["Discharge at C/20 until 2.5V (5 minute period)"])
sim = pybamm.Simulation(
    model=model, parameter_values=parameter_values, experiment=experiment
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
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options. As the scale of the parameters is large, a large sigma value
# is used to efficiently explore the parameter space
options = pybop.PintsOptions(sigma=20, max_iterations=250, max_unchanged_iterations=25)
optim = pybop.NelderMead(problem, options=options)

# Run optimisation
results = optim.run()
print(results)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Compare to known values
print("True parameters:", [parameter_values[p.name] for p in parameters])
print(f"Idenitified Parameters: {results.x}")

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_values = results.parameter_values

# Plot comparison
sim = pybamm.Simulation(
    model, parameter_values=identified_values, experiment=experiment
)
prediction = sim.solve()
pybop.plot.trajectories(
    x=[dataset["Time [s]"] / 3600, prediction.t / 3600],
    y=[dataset["Voltage [V]"], prediction["Voltage [V]"].data],
    trace_names=["Ground truth", "Identified model"],
    xaxis_title="Time / h",
    yaxis_title="Voltage / V",
)
