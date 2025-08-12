import numpy as np
import pybamm

import pybop

# In this example, we introduce the Maximum Likelihood
# Estimation (MLE) method. For time-series model identification
# MLE is computed for each observation and multiplied together.
# As the likelihood can be numerically small, PyBOP uses the log-likelihood.
# Likelihoods allow for incorporation of a noise model into the identification
# process as well as providing one of the core components in bayesian identification
# methods. MLE provides a point-based estimate of the parameter values, with
# a corresponding noise estimate if requested, as shown below.


# Define model and parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPMe()

# Construct the identification parameters
parameters = [
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.65,
        bounds=[0.4, 0.9],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.58,
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
options = pybop.PintsOptions(
    sigma=[0.01, 0.01, 0.01],
    verbose=True,
    max_iterations=100,
    max_unchanged_iterations=100,
)
optim = pybop.AdamW(problem, options=options)
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
