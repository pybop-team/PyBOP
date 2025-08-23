import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

# Define model and parameter values
model = pybop.ExponentialDecayModel()
parameter_values = pybamm.ParameterValues({"k": 0.3, "y0": 1.5})

# Fitting parameters
parameters = [
    pybop.Parameter("k", prior=pybop.Gaussian(0.5, 0.05), bounds=[0, 2]),
    pybop.Parameter("y0", prior=pybop.Gaussian(1.0, 0.05), bounds=[0, 2]),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 10, 50)
sol = sim.solve(t_eval=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "y_0": sol["y_0"](t_eval),
    }
)

# Construct the problem builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.MeanSquaredError("y_0", "y_0"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(sigma=0.2, verbose=True, max_iterations=100)
optim = pybop.CMAES(problem, options=options)
# optim.set_population_size(200)
# optim = pybop.SciPyMinimize(problem)
# optim = pybop.IRPropPlus(problem)
results = optim.run()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values
sim = pybamm.Simulation(model, parameter_values=identified_parameter_values)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)

# Plot the timeseries output
fig = go.Figure(layout=go.Layout(title="Time-domain comparison", width=800, height=600))

fig.add_trace(
    go.Scatter(
        x=dataset["Time [s]"],
        y=dataset["y_0"],
        mode="markers",
        name="Reference_0",
    )
)

fig.add_trace(go.Scatter(x=sol.t, y=sol["y_0"].data, mode="lines", name="y_0"))

fig.update_layout(
    xaxis_title="Time / s",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.show()

# Plot convergence
# pybop.plot.convergence(optim)

# Plot the parameter traces
# pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
# pybop.plot.contour(problem, steps=20)
