import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")

# The IDAKLU, and it's jaxified version perform very well on the DFN with and without
# gradient calculations
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
model = pybop.lithium_ion.SPM(parameter_set=parameter_set, solver=solver)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        initial_value=0.55,
        bounds=[0.5, 0.8],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.55,
        bounds=[0.5, 0.8],
    ),
)

# Define test protocol and generate data
t_eval = np.linspace(0, 600, 600)
values = model.predict(
    initial_state={"Initial open-circuit voltage [V]": 4.2}, t_eval=t_eval
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": values["Voltage [V]"].data,
    }
)

# Construct the Problem
problem = pybop.FittingProblem(model, parameters, dataset)

# By selecting a Jax based cost function, the IDAKLU solver will be
# jaxified (wrapped in a Jax compiled expression) and used for optimisation
cost = pybop.JaxSumSquaredError(problem)

# Non-gradient optimiser, change to `pybop.AdamW` for gradient-based example
optim = pybop.IRPropMin(
    cost,
    sigma0=0.02,
    max_unchanged_iterations=55,
    max_iterations=100,
)

results = optim.run()

# Plot convergence
pybop.plot.convergence(optim)

# Plot parameter trace
pybop.plot.parameters(optim)

# Plot voronoi optimiser surface
pybop.plot.surface(optim)
