import numpy as np
import pybamm

import pybop

# Define model and use high-performant solver for sensitivities
solver = pybamm.IDAKLUSolver()
parameter_set = pybop.ParameterSet("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.02),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.02),
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

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.Minkowski(problem, p=2)
optim = pybop.IRPropMin(cost, max_iterations=100, sigma0=0.1)

results = optim.run()

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Contour plot with optimisation path
bounds = np.asarray([[0.5, 0.8], [0.4, 0.7]])
pybop.plot.surface(optim, bounds=bounds)
