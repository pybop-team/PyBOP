import numpy as np
import pybamm

import pybop

"""
This example presents the process of creating a multi-fitting problem.
The multi-fitting problem allows for multiple problems to be optimised
at the same time, common use cases include:
- Fitting multiple datasets for a single model (varying SOC identification)
- Fitting different models for the same dataset (comparing reduced-order implementations)

Note: the optimisation parameters have to be the same for each problem.

In this example, we will identify parameters on the same model for two different datasets.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Create initial SOC and experiment objects
init_socs = [0.8, 0.6]
experiments = [
    pybamm.Experiment(["Discharge at 0.5C for 2 minutes (4 second period)"]),
    pybamm.Experiment(["Discharge at 1C for 1 minutes (4 second period)"]),
]

# Generate two fitting problems using synthetic data
sigma = 0.002
problems = []
for init_soc, experiment in zip(init_socs, experiments, strict=False):
    parameter_values.set_initial_state(init_soc)
    solution = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    ).solve()
    dataset = pybop.Dataset(
        {
            "Time [s]": solution.t,
            "Current function [A]": solution["Current [A]"].data,
            "Voltage [V]": solution["Voltage [V]"].data
            + np.random.normal(0, sigma, len(solution.t)),
        }
    )

    # Fitting parameters
    param_copy = parameter_values.copy()
    param_copy.update(
        {
            "Negative electrode active material volume fraction": pybop.Parameter(
                pybop.Gaussian(0.68, 0.05),
            ),
            "Positive electrode active material volume fraction": pybop.Parameter(
                pybop.Gaussian(0.58, 0.05),
            ),
        }
    )
    simulator = pybop.pybamm.Simulator(
        model, parameter_values=param_copy, protocol=dataset
    )
    cost = pybop.SumSquaredError(dataset)
    problems.append(pybop.Problem(simulator, cost))

# Combine the problems into one
problem = pybop.MetaProblem(*problems)

# Generate the cost function and optimisation class
options = pybop.PintsOptions(
    verbose=True,
    max_unchanged_iterations=20,
    max_iterations=100,
)
optim = pybop.CuckooSearch(problem, options=options)

# Run the optimisation
result = optim.run()

# Compare identified to true parameter values
print("True parameters:", [parameter_values[p] for p in problem.parameters.keys()])
print("Identified parameters:", result.x)

# Plot the timeseries output
pybop.plot.problem(problems[0], inputs=result.best_inputs, title="Optimised Comparison")
pybop.plot.problem(problems[1], inputs=result.best_inputs, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0.5, 0.8], [0.4, 0.7]])
