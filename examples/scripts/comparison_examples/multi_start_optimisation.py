import numpy as np
import pybamm
from scipy import stats

import pybop

"""
This example introduces the multi-start functionality in PyBOP. Multi-starting the
optimisation problem allows for an increased likelihood of converging to a global
optimal location within the search space. This is helpful for local optimisation
algorithms such as: Gradient Descent, AdamW, IRPropMin, ScipyMinimize, etc.
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
        "Negative electrode active material volume fraction": pybop.ParameterInfo(
            stats.norm(0.65, 0.1)
        ),
        "Positive electrode active material volume fraction": pybop.ParameterInfo(
            stats.norm(0.55, 0.1)
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(model, parameter_values, protocol=dataset)
cost = pybop.RootMeanSquaredError(dataset)
problem = pybop.Problem(simulator, cost)

# Construct the optimiser with multistart. Each of these runs has a random
# starting position sampled from the parameter priors
options = pybop.PintsOptions(max_iterations=150, multistart=5)
optim = pybop.GradientDescent(problem, options=options)
optim.optimiser.set_learning_rate(eta=0.02)

# Run the optimisation
result = optim.run()
print(result)

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0.5, 0.8], [0.4, 0.7]])
