import numpy as np
import pybamm

import pybop
from examples.standalone.model import ExponentialDecay

# Parameter set and model definition
parameter_set = pybamm.ParameterValues({"k": "[input]", "y0": "[input]"})
model = ExponentialDecay(parameter_set=parameter_set, n_states=1)
x0 = np.array([0.1, 1.0])

# Fitting parameters
parameters = [
    pybop.Parameter(
        "k",
        prior=pybop.Gaussian(0.1, 0.05),
        bounds=[0, 1],
    ),
    pybop.Parameter(
        "y0",
        prior=pybop.Gaussian(1, 0.05),
        bounds=[0, 3],
    ),
]

# Verification: save fixed inputs for testing
inputs = dict()
for i, param in enumerate(parameters):
    inputs[param.name] = x0[i]

# Make a prediction with measurement noise
sigma = 1e-2
t_eval = np.linspace(0, 20, 10)
values = model.predict(t_eval=t_eval, inputs=inputs)
values = values["2y"].data
corrupt_values = values + np.random.normal(0, sigma, len(t_eval))

# Verification step: compute the analytical solution for 2y
expected_values = 2 * inputs["y0"] * np.exp(-inputs["k"] * t_eval)

# Verification step: make another prediction using the Observer class
model.build(parameters=parameters)
simulator = pybop.Observer(parameters, model, signal=["2y"], x0=x0)
simulator._time_data = t_eval
measurements = simulator.evaluate(x0)

# Verification step: Compare by plotting
go = pybop.PlotlyManager().go
line1 = go.Scatter(x=t_eval, y=corrupt_values, name="Corrupt values", mode="markers")
line2 = go.Scatter(
    x=t_eval, y=expected_values, name="Expected trajectory", mode="lines"
)
line3 = go.Scatter(
    x=t_eval, y=measurements["2y"], name="Observed values", mode="markers"
)
fig = go.Figure(data=[line1, line2, line3])

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": 0 * t_eval,  # placeholder
        "2y": corrupt_values,
    }
)

# Build the model to get the number of states
model.build(dataset=dataset.data, parameters=parameters)

# Define the UKF observer
signal = ["2y"]
n_states = model.n_states
n_signals = len(signal)
covariance = np.diag([sigma**2] * n_states)
process_noise = np.diag([1e-6] * n_states)
measurement_noise = np.diag([sigma**2] * n_signals)
observer = pybop.UnscentedKalmanFilterObserver(
    parameters,
    model,
    covariance,
    process_noise,
    measurement_noise,
    dataset,
    signal=signal,
    x0=x0,
)

# Verification step: Find the maximum likelihood estimate given the true parameters
estimation = observer.evaluate(x0)

# Verification step: Add the estimate to the plot
line4 = go.Scatter(
    x=t_eval, y=estimation["2y"], name="Estimated trajectory", mode="lines"
)
fig.add_trace(line4)
fig.show()

# Generate problem, cost function, and optimisation class
cost = pybop.ObserverCost(observer)
optim = pybop.CMAES(cost, verbose=True)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output (requires model that returns Voltage)
pybop.quick_plot(observer, parameter_values=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot2d(optim, steps=15)
