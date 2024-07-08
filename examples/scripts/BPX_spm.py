import numpy as np

import pybop

# Define model
bpx_parameters = pybop.ParameterSet(
    json_path="examples/scripts/parameters/example_BPX.json"
)
parameter_set = bpx_parameters.import_from_bpx()
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative particle radius [m]",
        prior=pybop.Gaussian(6e-06, 0.1e-6),
        bounds=[1e-6, 9e-6],
        true_value=parameter_set["Negative particle radius [m]"],
    ),
    pybop.Parameter(
        "Positive particle radius [m]",
        prior=pybop.Gaussian(4.5e-07, 0.1e-6),
        bounds=[1e-7, 9e-7],
        true_value=parameter_set["Positive particle radius [m]"],
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 2)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, max_iterations=100)

# Run the optimisation
x, final_cost = optim.run()
print("True parameters:", parameters.true_value())
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path and updated bounds
pybop.plot2d(optim, steps=15)
