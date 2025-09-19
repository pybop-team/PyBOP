import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.1),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.1),
    ),
)

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

# Generate problem, cost function classes
simulator = pybop.pybamm.Simulator(
    model, parameter_values, input_parameter_names=parameters.names, protocol=dataset
)
problem = pybop.FittingProblem(simulator, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)

# Construct the optimiser with 10 multistart runs
# Each of these runs has a random starting position sampled
# from the parameter priors
options = pybop.PintsOptions(sigma=[0.6, 0.02], max_iterations=50, multistart=10)
optim = pybop.GradientDescent(cost, options=options)

# Run the optimisation
result = optim.run()
print(result)

# Plot the timeseries output, for the best run using the result.x attribute
pybop.plot.problem(problem, problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
pybop.plot.convergence(optim)
pybop.plot.parameters(optim)
pybop.plot.surface(optim, bounds=[[0.5, 0.8], [0.4, 0.7]])
