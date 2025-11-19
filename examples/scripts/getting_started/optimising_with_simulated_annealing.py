import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 0.001
t_eval = np.arange(0, 900, 3)
sol = pybamm.Simulation(model, parameter_values=parameter_values).solve(t_eval=t_eval)
corrupt_values = sol["Voltage [V]"](t_eval) + np.random.normal(0, sigma, len(t_eval))
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": sol["Current [A]"](t_eval),
        "Voltage [V]": corrupt_values,
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.6,
                0.1,
                truncated_at=[0.4, 0.85],
            ),
        ),
        "Positive electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(
                0.6,
                0.1,
                truncated_at=[0.4, 0.85],
            ),
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
    max_iterations=120,
    max_unchanged_iterations=60,
)
optim = pybop.SimulatedAnnealing(problem, options=options)

# Update initial temperature and cooling rate
# for the reduced number of iterations
optim.optimiser.temperature = 0.9
optim.optimiser.cooling_rate = 0.8

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0.5, 0.8], [0.4, 0.7]])
