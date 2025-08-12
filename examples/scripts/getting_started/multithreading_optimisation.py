import numpy as np
import pybamm

import pybop

# In this example, we will introduce Pybop's multithreading functionality
# for Pybamm based optimisation problems. This multithreading occurs
# within the pybop pipeline, specifically at the numerical solver level.
# To gain the most from this multithreading, Pints' population-based optimisers
# or samplers are recommended. This functionality is not currently supported
# for the Scipy optimisers.

# Set model, parameters
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Construct the dataset
t_eval = np.linspace(0, 100, 240)
sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
sol = sim.solve(t_eval=[t_eval[0], t_eval[-1]], t_interp=t_eval)
dataset = pybop.Dataset(
    {
        "Time [s]": sol.t,
        "Voltage [V]": sol["Voltage [V]"].data,
        "Current function [A]": sol["Current [A]"].data,
    }
)

# Create the builder
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(
        model,
        parameter_values=parameter_values,
    )
    .add_parameter(
        pybop.Parameter(
            "Negative electrode active material volume fraction",
            initial_value=0.6,
            transformation=pybop.LogTransformation(),
            bounds=[0.5, 0.8],
        )
    )
    .add_parameter(
        pybop.Parameter(
            "Positive electrode active material volume fraction",
            initial_value=0.6,
            transformation=pybop.LogTransformation(),
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

# To gain the most from the multithreading, we increase
# the population size for the optimiser. For the
# population-based optimiser, this results in a larger
# batch of parameter proposals for the numerical solver
# to parallelise.
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
