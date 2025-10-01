import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 2e-3
t_eval = np.linspace(0, 500, 240)
sol = pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval=t_eval)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)
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
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.6, 0.05),
            bounds=[0.5, 0.8],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            "Positive electrode active material volume fraction",
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
print("True values:", true_values)

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface()
