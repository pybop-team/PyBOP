import numpy as np
import pybamm

import pybop

"""
In this example, two parameters of an ECM model are identified. The model parameters are
reformulated so that the first branch capacitance C1 is defined in terms of the branch
resistance R1 and time constant Tau1. This allows us to identify R1 and Tau1 instead of
R1 and C1.
"""

# Define model and parameter values
model = pybamm.equivalent_circuit.Thevenin()
parameter_values = pybamm.ParameterValues("ECM_Example")
parameter_values.update(
    {
        "Initial SoC": 0.75,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": model.default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.002,
        "Element-1 initial overpotential [V]": 0,
        "R1 [Ohm]": 0.003,
        "C1 [F]": 2000,
    }
)

# PyBaMM wants to see capacitances, but it's better to fit time-constants, so let's introduce
# Tau1 to enable that
parameter_values.update(
    {
        "Tau1 [s]": parameter_values["R1 [Ohm]"] * parameter_values["C1 [F]"],
        "C1 [F]": pybamm.Parameter("Tau1 [s]") / pybamm.Parameter("R1 [Ohm]"),
    },
    check_already_exists=False,
)

# Define the parameters to fit
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.002, 0.001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "Tau1 [s]",
        prior=pybop.Gaussian(4.0, 0.2),
        bounds=[0, 9.0],
    ),
]

# Generate a synthetic dataset. When working with experimental observations, we wouldn't need
# to generate synthetic data, but here it gives us a known ground-truth to work with
experiment = pybamm.Experiment(
    [
        "Discharge at 1C for 2 minutes (2 second period)",
        "Rest for 1 minutes (2 second period)",
    ],
)
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sol = sim.solve()

sigma = 1e-4
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

# Set optimiser and options. We'll use the Nelder-Mead simplex based optimiser
options = pybop.PintsOptions(
    sigma=np.asarray([0.05, 0.5]),
    max_iterations=60,
    max_unchanged_iterations=15,
)
optim = pybop.NelderMead(problem, options=options)

# Run optimisation
results = optim.run()
print(results)

# Plot the parameter traces
optim.plot_parameters()

# Compare identified parameters with true parameters
print("True parameters:", [parameter_values[p.name] for p in parameters])
print("Estimated parameters:", results.x)

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, dataset=dataset)
