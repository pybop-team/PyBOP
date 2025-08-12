import numpy as np
import pybamm

import pybop

# In this example, we perform Maximum A Posteriori (MAP)
# identification. MAP allows for prior knowledge of the
# parameter values to be incorporated into the optimisation
# process. This is mathematically similar to performing
# regularized maximum likelihood estimation depending on
# the prior distribution selected.


# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPMe()

# We construct the fitting parameters with corresponding
# prior distributions that encapsulate our knowledge of the
# parameter values. PyBOP will then use these in the computation
# of the posterior distribution. When prior distributions are
# used, initial guesses are not required, as these are randomly
# generated from the distribution. Initial guesses can be used
# if you wish to override the random generation.
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

# Generate the synthetic dataset
sigma = 5e-3
t_eval = np.linspace(0, 500, 240)
sim = pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)

dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data
        + np.random.normal(0, sigma, len(t_eval)),
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Construct the problem builder with a negative
# Gaussian log-likelihood (NLL) function. Since we have
# not provided a `sigma` value to the NLL, this will be
# estimated from the data. `sigma` is the standard deviation
# of the measurement noise in the dataset.
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_cost(
        pybop.costs.pybamm.NegativeGaussianLogLikelihood("Voltage [V]", "Voltage [V]")
    )
)
for param in parameters:
    builder.add_parameter(param)
problem = builder.build()

# Set optimiser and options
options = pybop.SciPyDifferentialEvolutionOptions(verbose=True, maxiter=40)
optim = pybop.SciPyDifferentialEvolution(problem, options=options)
results = optim.run()

# Obtain the fully identified pybamm.ParameterValues object
# These can then be used with normal Pybamm classes
identified_parameter_values = results.parameter_values

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape with optimisation path
pybop.plot.contour(optim)
