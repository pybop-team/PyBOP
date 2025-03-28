import numpy as np
import pybamm

import pybop

# Define model and use the high-performant IDAKLU solver for sensitivities
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

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

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
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

# Generate problem, cost function classes
model.solver = pybamm.IDAKLUSolver(atol=1e-5, rtol=1e-5)
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.RootMeanSquaredError(problem)

# Construct the optimiser with 10 multistart runs
# Each of these runs has a random starting position sampled
# from the parameter priors
optim = pybop.GradientDescent(
    cost, sigma0=[0.6, 0.02], max_iterations=50, multistart=10, verbose=True
)

# Run optimisation
results = optim.run()

# We can plot the timeseries output, for the best run
# using the results.x attribute
pybop.plot.quick(problem, problem_inputs=results.x_best, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)
