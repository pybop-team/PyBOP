import numpy as np
import pybamm

import pybop

"""
In this example, we demonstrate identification using a Doyle-Fuller-Newman (DFN) model.
The DFN is more challenging to parameterise than equivalent circuit and reduced-order,
single particle models as it is more computationally expensive and has a high number of
parameters which are not individually identifiable.
"""

# Define model and parameter values
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
        "Bulk open-circuit voltage [V]": sol["Bulk open-circuit voltage [V]"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"))
    .add_cost(
        pybop.costs.pybamm.RootMeanSquaredError(
            "Bulk open-circuit voltage [V]", "Bulk open-circuit voltage [V]"
        )
    )
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options. We'll use the Nelder-Mead simplex based optimiser and increase the
# step-size value (sigma) to 0.1 to search further across the landscape per iteration
options = pybop.PintsOptions(
    sigma=0.02, max_iterations=250, max_unchanged_iterations=15
)
optim = pybop.NelderMead(problem, options=options)
results = optim.run()
print(results)

# Compare to known values
print("True parameters:", [parameter_values[p.name] for p in parameters])
print(f"Idenitified Parameters: {results.x}")

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

# Plot the cost landscape with optimisation path
optim.plot_surface()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_values = results.parameter_values

# Plot comparison
sim = pybamm.Simulation(model, parameter_values=identified_values)
prediction = sim.solve(t_eval=t_eval)
pybop.plot.trajectories(
    x=[dataset["Time [s]"], prediction.t],
    y=[dataset["Voltage [V]"], prediction["Voltage [V]"].data],
    trace_names=["Ground truth", "Identified model"],
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
)
