import numpy as np
import pybamm

import pybop

"""
In this example, we perform Maximum A Posteriori (MAP) identification. MAP allows for
prior knowledge of the parameter values to be incorporated into the optimisation process.
This is mathematically similar to performing regularized maximum likelihood estimation
depending on the prior distribution selected.

We construct the fitting parameters with corresponding prior distributions that
encapsulate our knowledge of the parameter values. PyBOP will use these in the
computation of the posterior distribution when the cost is an instance of
`pybop.LogLikelihood` and priors are provided. When prior distributions are used,
initial guesses are not required, as these are randomly generated from the distribution.
Initial guesses can be used if you wish to override the random generation.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
sigma = 5e-3
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

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Uniform(0.3, 0.8),
            bounds=[0.3, 0.8],
            initial_value=0.653,
            transformation=pybop.LogTransformation(),
        ),
        "Positive electrode active material volume fraction": pybop.Parameter(
            distribution=pybop.Uniform(0.3, 0.8),
            bounds=[0.4, 0.7],
            initial_value=0.657,
            transformation=pybop.LogTransformation(),
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
likelihood = pybop.GaussianLogLikelihood(dataset)
posterior = pybop.LogPosterior(likelihood)
problem = pybop.Problem(simulator, posterior)

# Set up the optimiser
options = pybop.PintsOptions(
    verbose=True,
    max_unchanged_iterations=20,
    min_iterations=20,
    max_iterations=50,
)
optim = pybop.XNES(problem, options=options)

# Run the optimisation
result = optim.run()

# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_contour(steps=10)
