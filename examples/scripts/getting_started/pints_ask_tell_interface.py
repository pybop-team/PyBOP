import numpy as np
import pybamm

import pybop

"""
In this example, we will introduce the ask-tell optimiser interface for the PINTS-based
optimisers. This interface provides a simple method for flexible optimisation workflows.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Generate a synthetic dataset
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
    t_eval=np.linspace(0, 100, 100)
)
dataset = pybop.Dataset(
    {
        "Time [s]": solution.t,
        "Current function [A]": solution["Current [A]"].data,
        "Voltage [V]": solution["Voltage [V]"].data,
        "Bulk open-circuit voltage [V]": solution["Bulk open-circuit voltage [V]"].data,
    }
)

# Fitting parameters
parameter_values.update(
    {
        "Negative electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(0.55, 0.05),
        ),
        "Positive electrode active material volume fraction": pybop.ParameterDistribution(
            distribution=pybop.Gaussian(0.55, 0.05),
        ),
    }
)

# Build the problem
simulator = pybop.pybamm.Simulator(
    model, parameter_values=parameter_values, protocol=dataset
)
cost = pybop.Minkowski(
    dataset, target=["Voltage [V]", "Bulk open-circuit voltage [V]"], p=2
)
problem = pybop.Problem(simulator, cost)

# We construct the optimiser class the same as normal but will be using the `optimiser`
# attribute directly for this example. This interface works for all PINTS-based
# optimisers. Warning: not all arguments are supported via this interface.
options = pybop.PintsOptions(verbose=True)
optim = pybop.AdamW(problem, options=options)

# Create storage vars
x_best = []
f_best = []

# Run the optimisation
for i in range(50):
    x = optim.optimiser.ask()
    f = [problem.evaluate(x[0], calculate_sensitivities=True).get_values()]
    optim.optimiser.tell(f)

    # Store best solution so far
    x_best.append(optim.optimiser.x_best())
    f_best.append(optim.optimiser.x_best())

    if i % 10 == 0:
        print(
            f"Iteration: {i} | Cost: {optim.optimiser.f_best()} | Parameters: {optim.optimiser.x_best()}"
        )

# Plot the timeseries output
pybop.plot.problem(
    problem,
    inputs=problem.parameters.to_dict(optim.optimiser.x_best()[0]),
    title="Optimised Comparison",
)
