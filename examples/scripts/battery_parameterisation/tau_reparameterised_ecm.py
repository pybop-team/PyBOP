import numpy as np
import pybamm
from matplotlib import pyplot as plt

import pybop

# In this example, an ECM is identified with parameters R0, and Tau1.
# The model parameters are formulated so that the first branch capacitance
# is linked to the branch resistance R1, and the time constant Tau1.
# This allows for a singular parameter to be fitted for each additional
# ECM branch, Tau`N`.


# Define the initial parameter set
parameter_values = pybamm.ParameterValues("ECM_Example")
parameter_values.update(
    {
        "Initial SoC": 0.75,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": pybamm.equivalent_circuit.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.002,
        "R1 [Ohm]": 0.003,
        "C1 [F]": 2000,
        "Element-1 initial overpotential [V]": 0,
    }
)


# PyBaMM wants to see capacitances, but it's better to fit
# time-constants, so let's introduce Tau1 to enable that
parameter_values.update(
    {
        "Tau1 [s]": parameter_values["R1 [Ohm]"] * parameter_values["C1 [F]"],
        "C1 [F]": pybamm.Parameter("Tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
    },
    check_already_exists=False,
)

# Define the model and parameters to fit
model = pybamm.equivalent_circuit.Thevenin()
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.002, 0.0001),
        transformation=pybop.LogTransformation(),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "Tau1 [s]",
        prior=pybop.Gaussian(4.0, 0.2),
        bounds=[0, 9.0],
    ),
]

# Generate a synthetic dataset for fitting. When working with
# experimental observations, we wouldn't need to generate synthetic data,
# but here it gives us a known ground-truth to work to.
experiment = pybamm.Experiment(
    [
        "Discharge at 1C for 2 minutes (2 second period)",
        "Rest for 1 minutes (2 second period)",
    ],
)
# Generate the synthetic dataset
sigma = 1e-4
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
    experiment=experiment,
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

# Set optimiser and options
# We use the Nelder-Mead simplex based
# optimiser for this example.
options = pybop.PintsOptions(
    sigma=np.asarray([0.05, 0.5]),
    verbose=True,
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.NelderMead(problem, options=options)
results = optim.run()

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Obtain the fully identified pybamm.ParameterValues and
# generate the time-series prediction using these parameters
identified_parameter_values = results.parameter_values
sim = pybamm.Simulation(
    model, parameter_values=identified_parameter_values, experiment=experiment
)
sol2 = sim.solve()

# Plot the time-series prediction and observations
fig, ax = plt.subplots()
ax.plot(sol2.t, sol2["Voltage [V]"].data, label="Fit")
ax.plot(dataset["Time [s]"], dataset["Voltage [V]"], label="Target")
ax.legend()
plt.show()


# Compare identified parameters with true parameters
print(
    "True parameters:",
    [
        parameter_values["R0 [Ohm]"],
        parameter_values["Tau1 [s]"],
    ],
)
print("Estimated parameters:", results.x)
