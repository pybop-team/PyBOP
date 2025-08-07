import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

# Import the ECM parameter set from JSON
# parameter_values = pybamm.ParameterValues.create_from_bpx(
#     "examples/parameters/initial_ecm_parameters.json"
# )

# Alternatively, define the initial parameter set with a dictionary
# Add definitions for R's, C's, and initial overpotentials for any additional RC elements
parameter_values = pybamm.ParameterValues(
    {
        "chemistry": "ecm",
        "Initial SoC": 0.5,
        "Initial temperature [K]": 25 + 273.15,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Ambient temperature [K]": 25 + 273.15,
        "Current function [A]": 5,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.0,
        "Cell thermal mass [J/K]": 1000,
        "Cell-jig heat transfer coefficient [W/K]": 10,
        "Jig thermal mass [J/K]": 500,
        "Jig-air heat transfer coefficient [W/K]": 10,
        "Open-circuit voltage [V]": pybamm.equivalent_circuit.Thevenin().default_parameter_values[
            "Open-circuit voltage [V]"
        ],
        "R0 [Ohm]": 0.001,
        "Element-1 initial overpotential [V]": 0,
        "Element-2 initial overpotential [V]": 0,
        "R1 [Ohm]": 0.0002,
        "R2 [Ohm]": 0.0003,
        "C1 [F]": 10000,
        "C2 [F]": 5000,
        "Entropic change [V/K]": 0.0004,
    }
)

# Define the model
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Fitting parameters
parameters = [
    pybop.Parameter(
        "R0 [Ohm]",
        prior=pybop.Gaussian(0.0002, 0.0001),
        bounds=[1e-4, 1e-2],
    ),
    pybop.Parameter(
        "R1 [Ohm]",
        prior=pybop.Gaussian(0.0001, 0.0001),
        bounds=[1e-5, 1e-2],
    ),
]


# Generate data
sigma = 0.001
t_eval = np.arange(0, 500, 3)
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

# Construct optimiser with additional options
options = pybop.PintsOptions(max_iterations=50)
optim = pybop.PSO(problem, options=options)

# Run optimisation
results = optim.run()

# Obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes
identified_parameter_values = results.parameter_values
sim = pybamm.Simulation(model, parameter_values=identified_parameter_values)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)

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

fig.add_trace(go.Scatter(x=sol.t, y=sol["Voltage [V]"].data, mode="lines", name="Fit"))

fig.update_layout(
    xaxis_title="Time / s",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.show()

# Plot the timeseries output
pybop.plot.surface(optim)
