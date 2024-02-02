import pybop
import pybamm
import numpy as np
from examples.standalone.exponential_decay import ExponentialDecay

# Parameter set and model definition
parameter_set = pybamm.ParameterValues({"k": "[input]", "y0": "[input]"})
model = ExponentialDecay(parameters=parameter_set, n_states=1)
model.build()

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

# Make a prediction without process noise
inputs = {"k": 0.1, "y0": 1.0}
signal = ["2y"]
t_eval = np.linspace(0, 20, 10)
sigma = 1e-2
values = model.predict(t_eval=t_eval, inputs=inputs)
values = values["2y"].data
corrupt_values = values + np.random.normal(0, sigma, len(t_eval))

# Verification step: make another prediction using the Observer class
simulator = pybop.Observer(model, inputs, signal)
measurements = []
for i, t in enumerate(t_eval):
    simulator.observe(t)
    ys = simulator.get_current_measure() + np.random.normal(0, sigma)
    measurements.append(ys)
measurements = np.hstack(measurements)[0]

# Verification step: compute the analytical solution for 2y
expected = 2 * inputs["y0"] * np.exp(-inputs["k"] * t_eval)

# Verification step: Compare by plotting
go = pybop.PlotlyManager().go
line1 = go.Scatter(x=t_eval, y=corrupt_values, name="Corrupt values", mode="markers")
line2 = go.Scatter(x=t_eval, y=measurements, name="Observed values", mode="markers")
line3 = go.Scatter(x=t_eval, y=expected, name="Expected tracjectory", mode="lines")
fig = go.Figure(data=[line1, line2, line3])

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": 0 * t_eval,
        signal[0]: corrupt_values,
    }
)

# Re-build model with dataset and unknown parameters
model.build(dataset=dataset, parameters=parameters)

# Define the UKF observer
n_states = model.n_states
n_signals = len(signal)
covariance = np.diag([sigma**2] * n_states)
process_noise = np.diag([1e-6] * n_states)
measurement_noise = np.diag([sigma**2] * n_signals)
observer = pybop.UnscentedKalmanFilterObserver(
    model, inputs, signal, covariance, process_noise, measurement_noise
)

# Verification step: Find the maximum likelihood estimate given the true parameters
ym = dataset.data[signal[0]]
estimation = []
for i, t in enumerate(t_eval):
    observer.observe(t, ym[i])
    ys = observer.get_current_measure()
    estimation.append(ys)
estimation = np.hstack(estimation)[0]

# Verification step: Add the estimate to the plot
line4 = go.Scatter(x=t_eval, y=estimation, name="Estimated trajectory", mode="lines")
fig.add_trace(line4)
fig.show()

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
cost = pybop.ObserverCost(problem, observer)
optim = pybop.Optimisation(cost, optimiser=pybop.CMAES, verbose=True)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output (requires model that returns Voltage)
pybop.quick_plot(x, cost, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape
pybop.plot_cost2d(cost, steps=15)

# Plot the cost landscape with optimisation path
pybop.plot_cost2d(cost, optim=optim, steps=15)
