import numpy as np
import pybamm

import pybop

"""
In this example, we introduce the Maximum Likelihood Estimation (MLE) method. For
time-series model identification MLE is computed for each observation and multiplied
together. As the likelihood can be numerically small, PyBOP uses the log-likelihood.
Likelihoods allow for incorporation of a noise model into the identification process
as well as providing one of the core components in Bayesian identification methods.
MLE provides a point-based estimate of the parameter values, with a corresponding
noise estimate if requested, as shown below.
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
        "Negative electrode active material volume fraction": pybop.ParameterInfo(
            distribution=pybop.Gaussian(
                0.6,
                0.05,
                truncated_at=[0.5, 0.8],
            )
        ),
        "Positive electrode active material volume fraction": pybop.ParameterInfo(
            distribution=pybop.Gaussian(0.48, 0.05),
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
likelihood = pybop.GaussianLogLikelihood(dataset, sigma0=8e-3)
problem = pybop.Problem(simulator, likelihood)

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
result.plot_contour(bounds=[[0.5, 0.8], [0.4, 0.7]], steps=10)
