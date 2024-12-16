import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybamm.ParameterValues({"k": 0.1, "y0": 1.0})
model = pybop.ExponentialDecayModel(parameter_set=parameter_set, n_states=1)

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


def noise(sigma):
    return np.random.normal(0, sigma, len(values["y_0"].data))


# Make a prediction with measurement noise
sigma = 1e-2
t_eval = np.linspace(0, 20, 10)
values = model.predict(t_eval=t_eval)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": 0 * t_eval,  # placeholder
        "y_0": values["y_0"].data + noise(sigma),
    }
)

# Build the model to get the number of states
model.build(dataset=dataset.data, parameters=parameters)

# Define the UKF observer
signal = ["y_0"]
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
