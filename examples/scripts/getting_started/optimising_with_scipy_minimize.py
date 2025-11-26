import numpy as np
import pybamm

import pybop

"""
In this example, we introduce the Scipy Minimize optimiser. This optimiser is a wrapper of
the scipy.optimize.minimize class, with options exposed where possible. Minimize offers both
gradient and non-gradient methods, where applicable the gradient of the cost is provided to
the optimiser.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 2e-3
t_eval = np.linspace(0, 500, 240)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=t_eval
)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": solution["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": solution["Current [A]"](t_eval),
    }
)

# Save the true values
true_values = [
    parameter_values[p]
    for p in [
        "Negative electrode active material volume fraction",
        "Positive electrode active material volume fraction",
    ]
]

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.6, 0.05),
            bounds=[0.5, 0.8],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            prior=pybop.Gaussian(0.48, 0.05),
            bounds=[0.4, 0.7],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.SumSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.SciPyMinimizeOptions(
    verbose=True, maxiter=100, method="L-BFGS-B", jac=True
)
optim = pybop.SciPyMinimize(problem, options=options)

# Run the optimisation
result = optim.run()

# Compare identified to true parameter values
print("True parameters:", true_values)
print("Identified parameters:", result.x)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
