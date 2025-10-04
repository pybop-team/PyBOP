import numpy as np
import pybamm

import pybop

# Define model and parameter values
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")

# Create initial SOC and experiment objects
init_socs = [0.8, 0.6]
experiments = [
    pybamm.Experiment([("Discharge at 0.5C for 2 minutes (4 second period)")]),
    pybamm.Experiment([("Discharge at 1C for 1 minutes (4 second period)")]),
]

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
    ),
)

# Generate two fitting problems using synthetic data
sigma = 0.002
problems = []
for init_soc, experiment in zip(init_socs, experiments, strict=False):
    parameter_values.set_initial_state(init_soc)
    sol = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    ).solve()
    dataset = pybop.Dataset(
        {
            "Time [s]": sol.t,
            "Current function [A]": sol["Current [A]"].data,
            "Voltage [V]": sol["Voltage [V]"].data
            + np.random.normal(0, sigma, len(sol.t)),
        }
    )
    simulator = pybop.pybamm.Simulator(
        model,
        parameter_values=parameter_values,
        parameters=parameters,
        protocol=dataset,
    )
    cost = pybop.SumSquaredError(dataset)
    problems.append(pybop.Problem(simulator, cost))

# Combine the problems into one
problem = pybop.MetaProblem(*problems)

# Generate the cost function and optimisation class
options = pybop.PintsOptions(
    verbose=True,
    sigma=0.05,
    max_unchanged_iterations=20,
    max_iterations=100,
)
optim = pybop.CuckooSearch(problem, options=options)

# Run the optimisation
result = optim.run()
print("True parameters:", [parameter_values[p] for p in parameters.keys()])

# Plot the timeseries output
pybop.plot.problem(problems[0], problem_inputs=result.x, title="Optimised Comparison")
pybop.plot.problem(problems[1], problem_inputs=result.x, title="Optimised Comparison")

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
result.plot_surface(bounds=[[0.5, 0.8], [0.4, 0.7]])
