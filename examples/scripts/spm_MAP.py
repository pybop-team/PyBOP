import numpy as np

import pybop

# Construct and update initial parameter values
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set.update(
    {
        "Negative electrode active material volume fraction": 0.63,
        "Positive electrode active material volume fraction": 0.51,
    }
)

# Define model
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.6, 0.05),
        bounds=[0.5, 0.8],
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.48, 0.05),
        bounds=[0.4, 0.7],
        true_value=parameter_set["Positive electrode active material volume fraction"],
    ),
)

# Generate data
sigma = 0.005
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
cost = pybop.MAP(problem, pybop.GaussianLogLikelihoodKnownSigma, sigma0=sigma)
optim = pybop.AdamW(
    cost,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=100,
)

# Run the optimisation
x, final_cost = optim.run()
print("True parameters:", parameters.true_value())
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x[0:2], title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape
pybop.plot2d(cost, steps=15)

# Plot the cost landscape with optimisation path
bounds = np.asarray([[0.55, 0.77], [0.48, 0.68]])
pybop.plot2d(optim, bounds=bounds, steps=15)
