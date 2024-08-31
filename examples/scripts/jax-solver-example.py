import time

import numpy as np
import pybamm

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")

# Set up the IDAKLU Solver, and enable jax compiliation via the pybop.model arg 'jax'
# The IDAKLU, and it's jaxified version perform very well on the DFN with and without
# gradient calulations
solver = pybamm.IDAKLUSolver()
model = pybop.lithium_ion.DFN(parameter_set=parameter_set, solver=solver, jax=True)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction", initial_value=0.55
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction", initial_value=0.55
    ),
)

# Define test protocol and generate data
t_eval = np.linspace(0, 100, 100)
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

problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.JaxLogNormalLikelihood(problem, sigma0=0.002)

# Non-gradient optimiser, change to `pybop.AdamW` for gradient-based example
optim = pybop.XNES(
    cost,
    max_unchanged_iterations=100,
    max_iterations=100,
)
optim.pints_optimiser.set_population_size(2)

start_time = time.time()
x = optim.run()
print(f"Total time: {time.time() - start_time}")
print(f"x:{x}")
print(f"{optim.result.n_iterations}")

pybop.plot_convergence(optim)
pybop.plot_parameters(optim)
