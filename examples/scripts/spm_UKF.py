import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.7],
    ),
]

# Make a prediction with measurement noise
sigma = 0.001
t_eval = np.arange(0, 300, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Build the model to get the number of states
model.build(dataset=dataset.data, parameters=parameters)

# Define the UKF observer, setting the particle boundaries as uncertain states
signal = ["Voltage [V]"]
n_states = model.n_states
n_signals = len(signal)
covariance = np.diag([0] * 19 + [sigma**2] + [0] * 19 + [sigma**2])
process_noise = np.diag([0] * 19 + [1e-6] + [0] * 19 + [1e-6])
measurement_noise = np.diag([sigma**2])
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
optim = pybop.PSO(cost, verbose=True)

# Parameter identification using the current observer implementation is very slow
# so let's restrict the number of iterations and reduce the number of plots
optim.set_max_iterations(5)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output (requires model that returns Voltage)
pybop.quick_plot(observer, parameter_values=x, title="Optimised Comparison")

# # Plot convergence
# pybop.plot_convergence(optim)

# # Plot the parameter traces
# pybop.plot_parameters(optim)

# # Plot the cost landscape with optimisation path
# pybop.plot2d(optim, steps=15)
