import numpy as np
import pybamm

import pybop

"""
In this example, we introduce the Maximum Likelihood Estimation (MLE) method. For
time-series model identification MLE is computed for each observation and multiplied
together. As the likelihood can be numerically small, PyBOP uses the log-likelihood.
Likelihoods allow for incorporation of a noise model into the identification process
as well as providing one of the core components in Bayesian identification methods.
MLE provides a point-based estimate of the parameter values, with a corresponding
noise estimate if requested, as shown below.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")

# Construct the identification parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.58,
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
    }
)

# Construct the problem builder with a negative Gaussian log-likelihood (NLL) function.
# Since we have not provided a `sigma` value to the NLL, this will be estimated from
# the data. `sigma` is the standard deviation of the measurement noise in the dataset.
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(
        pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]", "Voltage [V]")
    )
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.PintsOptions(
    sigma=[0.02, 0.02, 1e-4],
    verbose=True,
    max_iterations=100,
    max_unchanged_iterations=30,
)
optim = pybop.AdamW(problem, options=options)

# Run optimisation
results = optim.run()

# Plot convergence
optim.plot_convergence()

# Plot the parameter traces
optim.plot_parameters()

# Plot the cost landscape with optimisation path
optim.plot_contour()

# Obtain the identified pybamm.ParameterValues object for use with PyBaMM classes
identified_parameter_values = results.parameter_values

# Plot the timeseries output
sim = pybamm.Simulation(model, parameter_values=identified_parameter_values)
sol = sim.solve(t_eval=t_eval)

go = pybop.plot.PlotlyManager().go
fig = go.Figure(layout=go.Layout(xaxis_title="Time / s", yaxis_title="Voltage / V"))
fig.add_trace(
    go.Scatter(
        x=dataset["Time [s]"],
        y=dataset["Voltage [V]"],
        mode="markers",
        name="Target",
    )
)
fig.add_trace(
    go.Scatter(x=t_eval, y=sol["Voltage [V]"](t_eval), mode="lines", name="Fit")
)
fig.show()
