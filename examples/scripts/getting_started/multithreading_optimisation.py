import numpy as np
import pybamm

import pybop

"""
In this example, we will introduce PyBOP's multithreading functionality for PyBaMM-based
optimisation problems. This multithreading occurs within the pybop pipeline, specifically
at the numerical solver level. To gain the most from this multithreading, PINTS
population-based optimisers or samplers are recommended. This functionality is not currently
supported for the SciPy optimisers.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
t_eval = np.linspace(0, 100, 240)
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve(t_eval=t_eval)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Voltage [V]": sol["Voltage [V]"](t_eval),
        "Current function [A]": sol["Current [A]"](t_eval),
    }
)

# Create the builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_parameter(
        pybop.Parameter(
            "Negative electrode active material volume fraction",
            initial_value=0.6,
            bounds=[0.5, 0.8],
        )
    )
    .add_parameter(
        pybop.Parameter(
            "Positive electrode active material volume fraction",
            initial_value=0.6,
            bounds=[0.5, 0.8],
        )
    )
    .add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]"))
)

# Build the unconstrained multithreaded problem
problem = builder.build()

# Now let's update the problem with a limited number of threads
builder.set_n_threads(1)
problem2 = builder.build()

# To gain the most from the multithreading, we increase the population size for the
# optimiser. For the population-based optimiser, this results in a larger batch of
# parameter proposals for the numerical solver to parallelise
options = pybop.PintsOptions(max_iterations=5)
optim1 = pybop.CMAES(problem, options=options)
optim1.set_population_size(200)

optim2 = pybop.CMAES(problem2, options=options)
optim2.set_population_size(200)

# Run the first problem
results = optim1.run()
print(results)  # 0.640 seconds

# Run the second
results = optim2.run()
print(results)  # 0.896 seconds

# Surface plot of first optim
pybop.plot.surface(optim1)
pybop.plot.convergence(optim1)
pybop.plot.contour(problem)
