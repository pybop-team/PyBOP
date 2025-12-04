import numpy as np
import pybamm

import pybop

"""
In this example, we introduce Simulated Annealing, a probabilistic optimisation method
inspired by the annealing process in metallurgy. It works by iteratively proposing new
solutions and accepting them based on both their fitness and a temperature parameter that
decreases over time. This allows the algorithm to initially explore broadly and gradually
focus on local optimisation as the temperature decreases.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 0.001
t_eval = np.arange(0, 900, 3)
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=t_eval
)
corrupt_values = solution["Voltage [V]"](t_eval) + np.random.normal(
    0, sigma, len(t_eval)
)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": solution["Current [A]"](t_eval),
        "Voltage [V]": corrupt_values,
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.6, 0.1),
            bounds=[0.4, 0.85],
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Gaussian(0.6, 0.1),
            bounds=[0.4, 0.85],
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.RootMeanSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    max_iterations=150,
    max_unchanged_iterations=40,
)
optim = pybop.SimulatedAnnealing(problem, options=options)

# Update initial temperature and cooling rate to bias towards better solutions (lower exploration)
optim.optimiser.temperature = 0.01
optim.optimiser.cooling_rate = 0.9

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0.5, 0.8], [0.4, 0.7]])
