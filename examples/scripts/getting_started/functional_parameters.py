import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

# This example demonstrates how to use a pybamm.FunctionalParameter to
# optimise functional parameters using PyBOP.

# Method: Define a new scalar parameter for use in a functional parameter
# that already exists in the model, for example an exchange current density.


# Set parameter values
parameter_values = pybamm.ParameterValues("Chen2020")


# Define a new function using pybamm parameters
def positive_electrode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    # New parameters
    j0_ref = pybamm.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]"
    )
    alpha = pybamm.Parameter("Positive electrode charge transfer coefficient")

    # Existing parameters
    c_e_init = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        j0_ref
        * ((c_e / c_e_init) * (c_s_surf / c_s_max) * (1 - c_s_surf / c_s_max)) ** alpha
    )


# Give default values to the new scalar parameters and pass the new function
parameter_values.update(
    {
        "Positive electrode reference exchange-current density [A.m-2]": 1,
        "Positive electrode charge transfer coefficient": 0.5,
    },
    check_already_exists=False,
)
parameter_values["Positive electrode exchange-current density [A.m-2]"] = (
    positive_electrode_exchange_current_density
)

# Model definition
model = pybamm.lithium_ion.SPM(options={"contact resistance": "true"})

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Positive electrode reference exchange-current density [A.m-2]",
        prior=pybop.Gaussian(1, 0.1),
        bounds=[0.75, 1.25],
    ),
    pybop.Parameter(
        "Positive electrode charge transfer coefficient",
        prior=pybop.Gaussian(0.5, 0.1),
        bounds=[0.25, 0.75],
    ),
]

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)
corrupt_values = sol["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Current function [A]": sol["Current [A]"].data,
        "Voltage [V]": corrupt_values,
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

options = pybop.PintsOptions(sigma=0.1, max_iterations=125, verbose=True)
optim = pybop.NelderMead(problem, options=options)

# Run optimisation
results = optim.run()

# Plot the timeseries output
fig = go.Figure(layout=go.Layout(title="Time-domain comparison", width=800, height=600))

fig.add_trace(
    go.Scatter(
        x=dataset["Time [s]"],
        y=dataset["Voltage [V]"],
        mode="markers",
        name="Reference",
    )
)
fig.add_trace(
    go.Scatter(x=sol.t, y=sol["Voltage [V]"].data, mode="lines", name="Fitted")
)

fig.update_layout(
    xaxis_title="Time / s",
    yaxis_title="Voltage / V",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.show()

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
