import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model_1 = pybop.lithium_ion.SPM(parameter_set=parameter_set)
model_2 = pybop.lithium_ion.SPM(parameter_set=parameter_set.copy())

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        true_value=parameter_set["Positive electrode active material volume fraction"],
    ),
)

# Generate a dataset
sigma = 0.001
experiment_1 = pybop.Experiment([("Discharge at 0.5C for 2 minutes (4 second period)")])
values_1 = model_1.predict(experiment=experiment_1)
dataset_1 = pybop.Dataset(
    {
        "Time [s]": values_1["Time [s]"].data,
        "Current function [A]": values_1["Current [A]"].data,
        "Voltage [V]": values_1["Voltage [V]"].data
        + np.random.normal(0, sigma, len(values_1["Voltage [V]"].data)),
    }
)

# Generate a second dataset
experiment_2 = pybop.Experiment([("Discharge at 1C for 2 minutes (4 second period)")])
values_2 = model_2.predict(experiment=experiment_2)
dataset_2 = pybop.Dataset(
    {
        "Time [s]": values_2["Time [s]"].data,
        "Current function [A]": values_2["Current [A]"].data,
        "Voltage [V]": values_2["Voltage [V]"].data
        + np.random.normal(0, sigma, len(values_2["Voltage [V]"].data)),
    }
)

# Generate a problem for each dataset and combine into one
problem_1 = pybop.FittingProblem(model_1, parameters, dataset_1)
problem_2 = pybop.FittingProblem(model_2, parameters, dataset_2)
problem = pybop.MultiFittingProblem(problem_list=[problem_1, problem_2])

# Generate the cost function and optimisation class
cost = pybop.SumSquaredError(problem)
optim = pybop.IRPropMin(
    cost,
    # sigma0=0.011,
    verbose=True,
    max_iterations=12,
)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the timeseries output
pybop.quick_plot(problem_1, problem_inputs=x, title="Optimised Comparison")
pybop.quick_plot(problem_2, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot the cost landscape with optimisation path
bounds = np.array([[0.5, 0.8], [0.4, 0.7]])
pybop.plot2d(optim, bounds=bounds, steps=15)
