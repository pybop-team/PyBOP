import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet("Chen2020")

# The Jaxified IDAKLU performs very well on high iteration
# identification tasks, due to the just-in-time compilation
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.55,
        prior=pybop.Gaussian(0.6, 0.03),
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.55,
        prior=pybop.Gaussian(0.6, 0.03),
    ),
)

# Generate data
sigma = 0.002
experiment = pybamm.Experiment(
    [
        (
            "Charge at 0.5C for 3 minutes (3 second period)",
            "Discharge at 0.5C for 3 minutes (3 second period)",
        ),
    ]
)
values = model.predict(initial_state={"Initial SoC": 0.5}, experiment=experiment)


def noisy(data, sigma):
    return data + np.random.normal(0, sigma, len(data))


# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": noisy(values["Voltage [V]"].data, sigma),
    }
)

# Construct the Problem
problem = pybop.FittingProblem(model, parameters, dataset)

# By selecting a Jax based cost function, the IDAKLU solver will be
# jaxified (wrapped in a Jax compiled expression) and used for optimisation
cost = pybop.JaxLogNormalLikelihood(problem, sigma0=sigma)

# Test gradient-based optimiser
optim = pybop.IRPropMin(
    cost, sigma0=0.02, max_unchanged_iterations=35, max_iterations=100, verbose=True
)

results = optim.run()

# Plot convergence
pybop.plot.convergence(optim)

# Plot parameter trace
pybop.plot.parameters(optim)

# Plot voronoi surface
pybop.plot.surface(optim)
