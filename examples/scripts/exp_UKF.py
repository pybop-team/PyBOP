import numpy as np

import pybop
from examples.standalone.model import ExponentialDecay

# Parameter set and model definition
parameter_set = {"k": 0.1, "y0": 1.0}
model = ExponentialDecay(parameter_set=parameter_set, n_states=1)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "k",
        prior=pybop.Gaussian(0.1, 0.05),
        bounds=[0, 1],
        true_value=parameter_set["k"],
    ),
    pybop.Parameter(
        "y0",
        prior=pybop.Gaussian(1, 0.05),
        bounds=[0, 3],
        true_value=parameter_set["y0"],
    ),
)

# Make a prediction with measurement noise
sigma = 1e-2
t_eval = np.linspace(0, 20, 10)
true_inputs = parameters.as_dict("true")
values = model.predict(t_eval=t_eval)
values = values["2y"].data
corrupt_values = values + np.random.normal(0, sigma, len(t_eval))

# Verification step: compute the analytical solution for 2y
expected_values = (
    2 * parameters["y0"].true_value * np.exp(-parameters["k"].true_value * t_eval)
)

# Verification step: make another prediction using the Observer class
model.build(parameters=parameters)
simulator = pybop.Observer(parameters, model, signal=["2y"])
simulator.domain_data = t_eval
measurements = simulator.evaluate(true_inputs)

# Verification step: Compare by plot
go = pybop.plot.PlotlyManager().go
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
)

# Verification step: Find the maximum likelihood estimate given the true parameters
estimation = observer.evaluate(true_inputs)

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
results = optim.run()

# Plot the timeseries output (requires model that returns Voltage)
pybop.plot.quick(observer, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.surface(optim)
