import numpy as np

import pybop

# Parameter set and model definition
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPM(parameter_set=parameter_set)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Negative electrode active material volume fraction",
        prior=pybop.Gaussian(0.68, 0.05),
        bounds=[0.5, 0.8],
        true_value=parameter_set["Negative electrode active material volume fraction"],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        prior=pybop.Gaussian(0.58, 0.05),
        bounds=[0.4, 0.7],
        true_value=parameter_set["Positive electrode active material volume fraction"],
    ),
)

# Generate data
sigma = 0.001
t_eval = np.arange(0, 900, 3)
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
cost1 = pybop.SumSquaredError(problem)
cost2 = pybop.RootMeanSquaredError(problem)
weighted_cost = pybop.WeightedCost(cost_list=[cost1, cost2], weights=[1, 100])

for cost in [weighted_cost, cost1, cost2]:
    optim = pybop.IRPropMin(cost, max_iterations=60)

    # Run the optimisation
    x, final_cost = optim.run()
    print("True parameters:", parameters.true_value())
    print("Estimated parameters:", x)

    # Plot the timeseries output
    pybop.quick_plot(problem, parameter_values=x, title="Optimised Comparison")

    # Plot convergence
    pybop.plot_convergence(optim)

    # Plot the cost landscape with optimisation path
    pybop.plot2d(optim, steps=15)
