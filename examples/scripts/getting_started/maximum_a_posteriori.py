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
`pybop.BaseLikelihood` and priors are provided. When prior distributions are used,
initial guesses are not required, as these are randomly generated from the distribution.
Initial guesses can be used if you wish to override the random generation.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.4, 0.9],
    ),
]

# Generate a synthetic dataset
sim = pybamm.Simulation(model, parameter_values=parameter_values)
t_eval = np.linspace(0, 500, 240)
sol = sim.solve(t_eval=t_eval)

sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval)
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)

# Construct the problem builder with a negative Gaussian log-likelihood (NLL) function.
# Since we have not provided a `sigma` value to the NLL, this will be estimated from
# the data. `sigma` is the standard deviation of the measurement noise in the dataset.
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]"))
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.SciPyDifferentialEvolutionOptions(verbose=True, maxiter=40)
optim = pybop.SciPyDifferentialEvolution(problem, options=options)

# Run optimisation
results = optim.run()

# Plot convergence
results.plot_convergence()

# Plot the parameter traces
results.plot_parameters()

# Plot the cost landscape with optimisation path
results.plot_contour()

# Compare the fit to the data
pybop.plot.validation(results.x, problem=problem, dataset=dataset)
